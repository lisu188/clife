/*
 * MIT License
 *
 * Copyright (c) 2019 Andrzej Lis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "clife.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <ctime>
#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>


namespace {
    using Cell = LifeBoard::Cell;
    using CellSet = LifeBoard::CellSet;
    using CellBuffer = LifeBoard::CellBuffer;
    using RenderTarget = LifeBoard::RenderTarget;

    constexpr std::array<std::uint8_t, 18> kNextState = {
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
    };
    struct PixelPalette {
        std::uint32_t dead = 0;
        std::uint32_t alive = 0x00FFFFFFU;
    };

    struct alignas(32) PixelLookupTable {
        std::array<std::uint32_t, 256 * 8> entries{};
    };

    struct FreeDeleter {
        void operator()(std::uint8_t *data) const noexcept {
            std::free(data);
        }
    };

    using AlignedBytes = std::unique_ptr<std::uint8_t, FreeDeleter>;

    [[nodiscard]] std::uint32_t scale_component(std::uint8_t value, unsigned long mask) {
        if (mask == 0) {
            return 0;
        }

        const unsigned shift = static_cast<unsigned>(__builtin_ctzl(mask));
        const std::uint32_t max_value = static_cast<std::uint32_t>(mask >> shift);
        const std::uint32_t scaled = (static_cast<std::uint32_t>(value) * max_value + 127U) / 255U;
        return (scaled << shift) & static_cast<std::uint32_t>(mask);
    }

    [[nodiscard]] PixelPalette make_palette(const Visual *visual) {
        return {
                0U,
                scale_component(255U, visual->red_mask) |
                scale_component(255U, visual->green_mask) |
                scale_component(255U, visual->blue_mask),
        };
    }

    [[nodiscard]] PixelLookupTable make_pixel_lookup_table(std::uint32_t dead_pixel, std::uint32_t alive_pixel) {
        PixelLookupTable lut;
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            auto *entry = lut.entries.data() + pattern * 8;
            for (int bit = 0; bit < 8; ++bit) {
                entry[bit] = ((pattern >> bit) & 1U) != 0U ? alive_pixel : dead_pixel;
            }
        }
        return lut;
    }

    [[nodiscard]] AlignedBytes allocate_aligned_bytes(std::size_t size, std::size_t alignment) {
        const std::size_t aligned_size = (size + alignment - 1U) & ~(alignment - 1U);
        void *data = nullptr;
        if (posix_memalign(&data, alignment, aligned_size) != 0) {
            return AlignedBytes(nullptr);
        }
        return AlignedBytes(static_cast<std::uint8_t *>(data));
    }
#if defined(__i386__) || defined(__x86_64__)
    [[nodiscard]] bool detect_avx2() {
#if defined(__GNUC__) || defined(__clang__)
        static const bool supported = __builtin_cpu_supports("avx2");
        return supported;
#else
        return false;
#endif
    }

    __attribute__((target("avx2"), always_inline)) inline void step_block_avx2(
            const std::uint8_t *upper,
            const std::uint8_t *current,
            const std::uint8_t *lower,
            std::uint8_t *next,
            int x,
            const __m256i alive_value,
            const __m256i two_value,
            const __m256i three_value) {
        const __m256i upper_left = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(upper + x - 1));
        const __m256i upper_center = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(upper + x));
        const __m256i upper_right = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(upper + x + 1));
        const __m256i current_left = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(current + x - 1));
        const __m256i current_center = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(current + x));
        const __m256i current_right = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(current + x + 1));
        const __m256i lower_left = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lower + x - 1));
        const __m256i lower_center = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lower + x));
        const __m256i lower_right = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lower + x + 1));

        __m256i neighbors = _mm256_add_epi8(upper_left, upper_center);
        neighbors = _mm256_add_epi8(neighbors, upper_right);
        neighbors = _mm256_add_epi8(neighbors, current_left);
        neighbors = _mm256_add_epi8(neighbors, current_right);
        neighbors = _mm256_add_epi8(neighbors, lower_left);
        neighbors = _mm256_add_epi8(neighbors, lower_center);
        neighbors = _mm256_add_epi8(neighbors, lower_right);

        const __m256i alive_mask = _mm256_cmpeq_epi8(current_center, alive_value);
        const __m256i born_mask = _mm256_cmpeq_epi8(neighbors, three_value);
        const __m256i survive_mask = _mm256_and_si256(alive_mask, _mm256_cmpeq_epi8(neighbors, two_value));
        const __m256i next_mask = _mm256_or_si256(born_mask, survive_mask);
        const __m256i next_values = _mm256_and_si256(next_mask, alive_value);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(next + x), next_values);
    }

    __attribute__((target("avx2"))) [[nodiscard]] int step_row_avx2(
            const std::uint8_t *upper,
            const std::uint8_t *current,
            const std::uint8_t *lower,
            std::uint8_t *next,
            int view_width) {
        constexpr int kAvx2Width = 32;
        constexpr int kAvx2UnrolledWidth = kAvx2Width * 2;
        const __m256i alive_value = _mm256_set1_epi8(1);
        const __m256i two_value = _mm256_set1_epi8(2);
        const __m256i three_value = _mm256_set1_epi8(3);
        int x = 1;

        for (; (x + kAvx2UnrolledWidth - 1) <= view_width; x += kAvx2UnrolledWidth) {
            __builtin_prefetch(upper + x + (kAvx2UnrolledWidth * 2), 0, 1);
            __builtin_prefetch(current + x + (kAvx2UnrolledWidth * 2), 0, 1);
            __builtin_prefetch(lower + x + (kAvx2UnrolledWidth * 2), 0, 1);
            __builtin_prefetch(next + x + (kAvx2UnrolledWidth * 2), 1, 1);
            step_block_avx2(upper, current, lower, next, x, alive_value, two_value, three_value);
            step_block_avx2(upper, current, lower, next, x + kAvx2Width, alive_value, two_value, three_value);
        }

        for (; (x + kAvx2Width - 1) <= view_width; x += kAvx2Width) {
            step_block_avx2(upper, current, lower, next, x, alive_value, two_value, three_value);
        }

        return x;
    }

    __attribute__((target("avx2"), always_inline)) inline void store_pixels_avx2(
            std::uint32_t *pixel_row,
            int x,
            const std::uint32_t *pixel_lut,
            unsigned pattern,
            bool stream_stores) {
        const auto *entry = reinterpret_cast<const __m256i *>(
                pixel_lut + static_cast<std::size_t>(pattern) * 8U);
        const __m256i colored = _mm256_load_si256(entry);
        if (stream_stores) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(pixel_row + x), colored);
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(pixel_row + x), colored);
        }
    }

    __attribute__((target("avx2"), always_inline)) inline void paint_block_avx2_lut(
            const std::uint8_t *cell_row,
            std::uint32_t *pixel_row,
            int x,
            const std::uint32_t *pixel_lut,
            bool stream_stores) {
        const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cell_row + x));
        const __m256i high_bits = _mm256_slli_epi16(source, 7);
        const unsigned packed_mask = static_cast<unsigned>(_mm256_movemask_epi8(high_bits));
        store_pixels_avx2(pixel_row, x, pixel_lut, packed_mask & 0xFFU, stream_stores);
        store_pixels_avx2(pixel_row, x + 8, pixel_lut, (packed_mask >> 8) & 0xFFU, stream_stores);
        store_pixels_avx2(pixel_row, x + 16, pixel_lut, (packed_mask >> 16) & 0xFFU, stream_stores);
        store_pixels_avx2(pixel_row, x + 24, pixel_lut, (packed_mask >> 24) & 0xFFU, stream_stores);
    }

    __attribute__((target("avx2"))) void paint_row_avx2(
            const std::uint8_t *cell_row,
            std::uint32_t *pixel_row,
            int view_width,
            const std::uint32_t *pixel_lut,
            bool stream_stores) {
        constexpr int kRenderBlockWidth = 32;
        constexpr int kRenderUnrolledWidth = 64;

        int x = 0;
        for (; (x + kRenderUnrolledWidth) <= view_width; x += kRenderUnrolledWidth) {
            __builtin_prefetch(cell_row + x + (kRenderUnrolledWidth * 2), 0, 1);
            paint_block_avx2_lut(cell_row, pixel_row, x, pixel_lut, stream_stores);
            paint_block_avx2_lut(cell_row, pixel_row, x + kRenderBlockWidth, pixel_lut, stream_stores);
        }

        for (; (x + kRenderBlockWidth) <= view_width; x += kRenderBlockWidth) {
            paint_block_avx2_lut(cell_row, pixel_row, x, pixel_lut, stream_stores);
        }

        for (; x < view_width; ++x) {
            pixel_row[x] = pixel_lut[static_cast<std::size_t>(cell_row[x]) * 8U];
        }
    }
#endif

    inline void paint_row_scalar(
            const std::uint8_t *cell_row,
            std::uint32_t *pixel_row,
            int view_width,
            std::uint32_t dead_pixel,
            std::uint32_t alive_pixel) {
        for (int x = 0; x < view_width; ++x) {
            pixel_row[x] = cell_row[x] != 0 ? alive_pixel : dead_pixel;
        }
    }

    void paint_rows(
            const std::uint8_t *cells,
            const RenderTarget &render_target,
            int begin_row,
            int end_row,
            int view_width,
            int stride,
            int top_left_index) {
        std::uint8_t *surface_bytes = render_target.surface_bytes;
        const int pitch_bytes = render_target.pitch_bytes;
        const std::uint32_t dead_pixel = render_target.dead_pixel;
        const std::uint32_t alive_pixel = render_target.alive_pixel;
#if defined(__i386__) || defined(__x86_64__)
        if (render_target.use_avx2) {
            for (int row = begin_row; row < end_row; ++row) {
                const std::uint8_t *cell_row = cells + top_left_index + row * stride;
                auto *pixel_row = reinterpret_cast<std::uint32_t *>(
                        surface_bytes + static_cast<std::ptrdiff_t>(row) * pitch_bytes);
                paint_row_avx2(cell_row,
                               pixel_row,
                               view_width,
                               render_target.pixel_lut,
                               render_target.stream_stores);
            }
            if (render_target.stream_stores) {
                _mm_sfence();
            }
            return;
        }
#endif
        for (int row = begin_row; row < end_row; ++row) {
            const std::uint8_t *cell_row = cells + top_left_index + row * stride;
            auto *pixel_row = reinterpret_cast<std::uint32_t *>(
                    surface_bytes + static_cast<std::ptrdiff_t>(row) * pitch_bytes);
            paint_row_scalar(cell_row, pixel_row, view_width, dead_pixel, alive_pixel);
        }
    }
#if defined(__SSE2__)
    constexpr int kSimdWidth = 16;
#endif

    int infer_extent(const std::shared_ptr<CellSet> &board, int requested, bool x_axis) {
        if (requested > 0) {
            return requested;
        }

        int inferred = 1;
        if (!board) {
            return inferred;
        }

        for (const Cell &cell: *board) {
            const int value = x_axis ? cell.first : cell.second;
            if (value >= 0) {
                inferred = std::max(inferred, value + 1);
            }
        }
        return inferred;
    }

    [[nodiscard]] int wrap_coordinate(int value, int extent) {
        const int wrapped = value % extent;
        return wrapped >= 0 ? wrapped : wrapped + extent;
    }

    [[nodiscard]] std::uint64_t mix_seed(std::uint64_t seed) {
        seed += 0x9E3779B97F4A7C15ULL;
        seed = (seed ^ (seed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        seed = (seed ^ (seed >> 27U)) * 0x94D049BB133111EBULL;
        return seed ^ (seed >> 31U);
    }

    class FastRng {
    public:
        explicit FastRng(std::uint64_t seed)
                : _state(seed == 0 ? 0xA5A5A5A5A5A5A5A5ULL : seed) {
        }

        [[nodiscard]] std::uint64_t next() {
            std::uint64_t value = _state;
            value ^= value >> 12U;
            value ^= value << 25U;
            value ^= value >> 27U;
            _state = value;
            return value * 2685821657736338717ULL;
        }

    private:
        std::uint64_t _state;
    };

    [[nodiscard]] CellBuffer make_seed_cells(int width, int height, float density) {
        const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        CellBuffer cells(total, 0);

        if (density <= 0.0f) {
            return cells;
        }
        if (density >= 1.0f) {
            std::fill(cells.begin(), cells.end(), static_cast<std::uint8_t>(1));
            return cells;
        }

        const auto threshold = static_cast<std::uint64_t>(
                std::clamp(static_cast<double>(density), 0.0, 1.0) *
                (static_cast<double>(std::numeric_limits<std::uint32_t>::max()) + 1.0));
        FastRng rng(mix_seed(static_cast<std::uint64_t>(std::time(nullptr))) ^
                    mix_seed(static_cast<std::uint64_t>(width)) ^
                    (mix_seed(static_cast<std::uint64_t>(height)) << 1U));
        std::size_t index = 0;
        for (; (index + 1) < total; index += 2) {
            const std::uint64_t random_pair = rng.next();
            cells[index] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(random_pair) < threshold);
            cells[index + 1] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(random_pair >> 32U) < threshold);
        }
        if (index < total) {
            cells[index] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(rng.next()) < threshold);
        }
        return cells;
    }

    struct TaskRange {
        int begin_row = 0;
        int end_row = 0;
    };

    class ThreadPool {
    public:
        explicit ThreadPool(std::size_t background_threads) {
            _threads.reserve(background_threads);
            for (std::size_t index = 0; index < background_threads; ++index) {
                _threads.emplace_back(&ThreadPool::worker_loop, this);
            }
        }

        ~ThreadPool() {
            if (_threads.empty()) {
                return;
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stop = true;
                ++_generation;
            }
            _work_cv.notify_all();

            for (std::thread &thread: _threads) {
                thread.join();
            }
        }

        template<typename Task>
        void parallel_for(std::size_t task_count, Task &&task) {
            if (task_count == 0) {
                return;
            }

            using TaskType = std::decay_t<Task>;
            TaskType task_storage(std::forward<Task>(task));
            if (_threads.empty() || (task_count == 1)) {
                for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
                    task_storage(task_index);
                }
                return;
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                _task_context = &task_storage;
                _task_entry = [](void *context, std::size_t task_index) {
                    (*static_cast<TaskType *>(context))(task_index);
                };
                _task_count = task_count;
                _next_task.store(0, std::memory_order_relaxed);
                _completed_workers = 0;
                ++_generation;
            }
            _work_cv.notify_all();

            run_pending_tasks();

            std::unique_lock<std::mutex> lock(_mutex);
            _done_cv.wait(lock, [this]() {
                return _completed_workers == _threads.size();
            });
            _task_entry = nullptr;
            _task_context = nullptr;
        }

    private:
        void run_pending_tasks() {
            while (true) {
                const std::size_t task_index = _next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_index >= _task_count) {
                    return;
                }
                _task_entry(_task_context, task_index);
            }
        }

        void worker_loop() {
            std::size_t seen_generation = 0;
            std::unique_lock<std::mutex> lock(_mutex);
            while (true) {
                _work_cv.wait(lock, [this, &seen_generation]() {
                    return _stop || (_generation != seen_generation);
                });

                if (_stop) {
                    return;
                }

                seen_generation = _generation;
                lock.unlock();
                run_pending_tasks();
                lock.lock();

                ++_completed_workers;
                if (_completed_workers == _threads.size()) {
                    _done_cv.notify_one();
                }
            }
        }

        std::vector<std::thread> _threads;
        std::mutex _mutex;
        std::condition_variable _work_cv;
        std::condition_variable _done_cv;
        void (*_task_entry)(void *, std::size_t) = nullptr;
        void *_task_context = nullptr;
        std::atomic<std::size_t> _next_task = 0;
        std::size_t _task_count = 0;
        std::size_t _generation = 0;
        std::size_t _completed_workers = 0;
        bool _stop = false;
    };

    struct HeadlessFrameBuffer {
        AlignedBytes bytes;
        int pitch_bytes = 0;
        PixelPalette palette{};
        bool stream_stores = false;
    };

    class X11FrameBuffer {
    public:
        X11FrameBuffer() = default;

        ~X11FrameBuffer() {
            destroy();
        }

        X11FrameBuffer(const X11FrameBuffer &) = delete;
        X11FrameBuffer &operator=(const X11FrameBuffer &) = delete;

        [[nodiscard]] bool create(int width, int height) {
            _display = XOpenDisplay(nullptr);
            if (_display == nullptr) {
                std::cerr << "XOpenDisplay failed" << std::endl;
                return false;
            }

            const int screen = DefaultScreen(_display);
            Visual *visual = DefaultVisual(_display, screen);
            const int depth = DefaultDepth(_display, screen);
            _palette = make_palette(visual);

            _window = XCreateSimpleWindow(_display,
                                          RootWindow(_display, screen),
                                          0,
                                          0,
                                          static_cast<unsigned>(width),
                                          static_cast<unsigned>(height),
                                          0,
                                          0,
                                          0);
            if (_window == 0) {
                std::cerr << "XCreateSimpleWindow failed" << std::endl;
                return false;
            }

            _gc = XCreateGC(_display, _window, 0, nullptr);
            if (_gc == nullptr) {
                std::cerr << "XCreateGC failed" << std::endl;
                return false;
            }

            _wm_delete = XInternAtom(_display, "WM_DELETE_WINDOW", False);
            XSetWMProtocols(_display, _window, &_wm_delete, 1);
            XSelectInput(_display, _window, ExposureMask | StructureNotifyMask | KeyPressMask);
            XMapWindow(_display, _window);
            XSync(_display, False);

            if (XShmQueryExtension(_display)) {
                _image = XShmCreateImage(_display, visual, static_cast<unsigned>(depth), ZPixmap, nullptr, &_shm_info, width, height);
                if ((_image != nullptr) && (_image->bits_per_pixel == 32)) {
                    const std::size_t buffer_size =
                            static_cast<std::size_t>(_image->bytes_per_line) * static_cast<std::size_t>(_image->height);
                    _shm_info.shmid = shmget(IPC_PRIVATE, buffer_size, IPC_CREAT | 0600);
                    if (_shm_info.shmid >= 0) {
                        _shm_info.shmaddr = static_cast<char *>(shmat(_shm_info.shmid, nullptr, 0));
                        if (_shm_info.shmaddr != reinterpret_cast<char *>(-1)) {
                            _image->data = _shm_info.shmaddr;
                            _shm_info.readOnly = False;
                            if (XShmAttach(_display, &_shm_info) != 0) {
                                XSync(_display, False);
                                shmctl(_shm_info.shmid, IPC_RMID, nullptr);
                                _use_shm = true;
                            } else {
                                shmdt(_shm_info.shmaddr);
                                _shm_info.shmaddr = nullptr;
                            }
                        }
                    }
                }
                if (!_use_shm && (_image != nullptr)) {
                    _image->data = nullptr;
                    XDestroyImage(_image);
                    _image = nullptr;
                }
            }

            if (_image == nullptr) {
                _image = XCreateImage(_display, visual, static_cast<unsigned>(depth), ZPixmap, 0, nullptr, width, height, 32, 0);
                if ((_image == nullptr) || (_image->bits_per_pixel != 32)) {
                    std::cerr << "XCreateImage failed or returned an unsupported format" << std::endl;
                    return false;
                }

                const std::size_t buffer_size =
                        static_cast<std::size_t>(_image->bytes_per_line) * static_cast<std::size_t>(_image->height);
                AlignedBytes data = allocate_aligned_bytes(buffer_size, 64);
                if (!data) {
                    std::cerr << "Failed to allocate X11 image buffer" << std::endl;
                    return false;
                }
                std::memset(data.get(), 0, buffer_size);
                _image->data = reinterpret_cast<char *>(data.release());
            }

            _pitch_bytes = _image->bytes_per_line;
            _stream_stores =
                    ((static_cast<std::uintptr_t>(_pitch_bytes) | reinterpret_cast<std::uintptr_t>(_image->data)) & 31U) == 0U;
            return true;
        }

        void destroy() {
            if (_display != nullptr) {
                if (_use_shm && (_image != nullptr)) {
                    XShmDetach(_display, &_shm_info);
                    if (_shm_info.shmaddr != nullptr) {
                        shmdt(_shm_info.shmaddr);
                        _shm_info.shmaddr = nullptr;
                    }
                    _image->data = nullptr;
                }
                if (_image != nullptr) {
                    XDestroyImage(_image);
                    _image = nullptr;
                }
                if (_gc != nullptr) {
                    XFreeGC(_display, _gc);
                    _gc = nullptr;
                }
                if (_window != 0) {
                    XDestroyWindow(_display, _window);
                    _window = 0;
                }
                XCloseDisplay(_display);
                _display = nullptr;
            }
        }

        void poll_events(bool &running) {
            while ((_display != nullptr) && (XPending(_display) > 0)) {
                XEvent event;
                XNextEvent(_display, &event);
                if ((event.type == ClientMessage) &&
                    (static_cast<Atom>(event.xclient.data.l[0]) == _wm_delete)) {
                    running = false;
                } else if (event.type == DestroyNotify) {
                    running = false;
                }
            }
        }

        void present(int width, int height) {
            if (_use_shm) {
                XShmPutImage(_display, _window, _gc, _image, 0, 0, 0, 0, static_cast<unsigned>(width), static_cast<unsigned>(height), False);
            } else {
                XPutImage(_display, _window, _gc, _image, 0, 0, 0, 0, static_cast<unsigned>(width), static_cast<unsigned>(height));
            }
            XFlush(_display);
        }

        [[nodiscard]] std::uint8_t *pixels() const {
            return reinterpret_cast<std::uint8_t *>(_image->data);
        }

        [[nodiscard]] int pitch_bytes() const {
            return _pitch_bytes;
        }

        [[nodiscard]] const PixelPalette &palette() const {
            return _palette;
        }

        [[nodiscard]] bool stream_stores() const {
            return _stream_stores;
        }

    private:
        Display *_display = nullptr;
        Window _window = 0;
        GC _gc = nullptr;
        Atom _wm_delete = 0;
        XImage *_image = nullptr;
        XShmSegmentInfo _shm_info{};
        PixelPalette _palette{};
        int _pitch_bytes = 0;
        bool _use_shm = false;
        bool _stream_stores = false;
    };

    [[nodiscard]] HeadlessFrameBuffer create_headless_frame_buffer(int width, int height) {
        HeadlessFrameBuffer buffer;
        buffer.pitch_bytes = width * static_cast<int>(sizeof(std::uint32_t));
        buffer.bytes = allocate_aligned_bytes(static_cast<std::size_t>(buffer.pitch_bytes) * static_cast<std::size_t>(height), 64);
        buffer.palette = {0U, 0x00FFFFFFU};
        buffer.stream_stores =
                buffer.bytes &&
                (((static_cast<std::uintptr_t>(buffer.pitch_bytes) | reinterpret_cast<std::uintptr_t>(buffer.bytes.get())) & 31U) == 0U);
        return buffer;
    }
}

class LifeBoard::Impl {
public:
    Impl(std::shared_ptr<CellSet> board, int threads, int width, int height)
            : _view_width(infer_extent(board, width, true)),
              _view_height(infer_extent(board, height, false)),
              _stride(_view_width + 2),
              _parallelism(std::max(1, std::min(std::max(1, threads), _view_height))),
#if defined(__i386__) || defined(__x86_64__)
              _use_avx2(detect_avx2()),
#else
              _use_avx2(false),
#endif
              _thread_pool(static_cast<std::size_t>(std::max(0, _parallelism - 1))),
              _front(static_cast<std::size_t>(_stride) * static_cast<std::size_t>(_view_height + 2), 0),
              _back(_front.size(), 0),
              _task_ranges(static_cast<std::size_t>(_parallelism)) {
        initialize_ranges();

        if (!board) {
            update_halo(_front);
            return;
        }

        for (const Cell &cell: *board) {
            const int index = to_index(wrap_coordinate(cell.first, _view_width),
                                       wrap_coordinate(cell.second, _view_height));
            std::uint8_t &slot = _front[static_cast<std::size_t>(index)];
            if (slot != 0) {
                continue;
            }
            slot = 1;
        }

        update_halo(_front);
    }

    Impl(CellBuffer cells, int threads, int width, int height)
            : _view_width(std::max(1, width)),
              _view_height(std::max(1, height)),
              _stride(_view_width + 2),
              _parallelism(std::max(1, std::min(std::max(1, threads), _view_height))),
#if defined(__i386__) || defined(__x86_64__)
              _use_avx2(detect_avx2()),
#else
              _use_avx2(false),
#endif
              _thread_pool(static_cast<std::size_t>(std::max(0, _parallelism - 1))),
              _front(static_cast<std::size_t>(_stride) * static_cast<std::size_t>(_view_height + 2), 0),
              _back(_front.size(), 0),
              _task_ranges(static_cast<std::size_t>(_parallelism)) {
        initialize_ranges();

        const std::size_t required = static_cast<std::size_t>(_view_width) * static_cast<std::size_t>(_view_height);
        if (cells.size() < required) {
            cells.resize(required, 0);
        }

        for (int row = 0; row < _view_height; ++row) {
            const std::size_t src_index = static_cast<std::size_t>(row) * static_cast<std::size_t>(_view_width);
            const int dst_index = row_start(row);
            auto src = cells.begin() + static_cast<std::ptrdiff_t>(src_index);
            auto dst = _front.begin() + dst_index;
            std::copy_n(src, _view_width, dst);
        }
        update_halo(_front);
    }

    [[nodiscard]] LifeBoard::FrameView iterate(const RenderTarget *render_target) {
        if (_first_frame) {
            _first_frame = false;
            if (render_target != nullptr) {
                _thread_pool.parallel_for(static_cast<std::size_t>(_parallelism), [this, render_target](std::size_t task_index) {
                    paint_current_chunk(task_index, *render_target);
                });
            }
            return {&_front, _view_width, _view_height, _stride, _stride + 1};
        }

        update_halo(_front);

        _thread_pool.parallel_for(static_cast<std::size_t>(_parallelism), [this, render_target](std::size_t task_index) {
            evaluate_chunk(task_index, render_target);
        });

        _front.swap(_back);

        return {&_front, _view_width, _view_height, _stride, _stride + 1};
    }

    [[nodiscard]] CellSet snapshot() const {
        CellSet board;
        for (int y = 0; y < _view_height; ++y) {
            int index = row_start(y);
            for (int x = 0; x < _view_width; ++x, ++index) {
                if (_front[static_cast<std::size_t>(index)] != 0) {
                    board.emplace(x, y);
                }
            }
        }
        return board;
    }

    [[nodiscard]] bool alive(Cell cell) const {
        return _front[static_cast<std::size_t>(to_index(wrap_coordinate(cell.first, _view_width),
                                                        wrap_coordinate(cell.second, _view_height)))] != 0;
    }

    [[nodiscard]] int width() const {
        return _view_width;
    }

    [[nodiscard]] int height() const {
        return _view_height;
    }

private:
    void initialize_ranges() {
        const int rows_per_chunk = _view_height / _parallelism;
        const int extra_rows = _view_height % _parallelism;
        for (int task = 0; task < _parallelism; ++task) {
            const int begin_row = (task * rows_per_chunk) + std::min(task, extra_rows);
            const int row_count = rows_per_chunk + (task < extra_rows ? 1 : 0);
            _task_ranges[static_cast<std::size_t>(task)] = {begin_row, begin_row + row_count};
        }
    }

    [[nodiscard]] int row_start(int row) const {
        return (row + 1) * _stride + 1;
    }

    [[nodiscard]] int to_index(int x, int y) const {
        return row_start(y) + x;
    }

    void update_halo(std::vector<std::uint8_t> &cells) const {
        for (int row = 0; row < _view_height; ++row) {
            const int row_index = (row + 1) * _stride;
            cells[static_cast<std::size_t>(row_index)] =
                    cells[static_cast<std::size_t>(row_index + _view_width)];
            cells[static_cast<std::size_t>(row_index + _view_width + 1)] =
                    cells[static_cast<std::size_t>(row_index + 1)];
        }

        const int first_visible_row = _stride;
        const int last_visible_row = _view_height * _stride;
        std::copy_n(cells.begin() + last_visible_row, _stride, cells.begin());
        std::copy_n(cells.begin() + first_visible_row,
                    _stride,
                    cells.begin() + static_cast<std::ptrdiff_t>((_view_height + 1) * _stride));
    }

    void paint_current_chunk(std::size_t task_index, const RenderTarget &render_target) const {
        const TaskRange range = _task_ranges[task_index];
        if (range.begin_row >= range.end_row) {
            return;
        }

        paint_rows(_front.data(), render_target, range.begin_row, range.end_row, _view_width, _stride, _stride + 1);
    }

    void evaluate_chunk(std::size_t task_index, const RenderTarget *render_target) {
        const TaskRange range = _task_ranges[task_index];
        if (range.begin_row >= range.end_row) {
            return;
        }

        const std::uint8_t *front = _front.data();
        std::uint8_t *back = _back.data();

        for (int y = range.begin_row; y < range.end_row; ++y) {
            const std::uint8_t *upper = front + static_cast<std::ptrdiff_t>(y * _stride);
            const std::uint8_t *current = upper + _stride;
            const std::uint8_t *lower = current + _stride;
            std::uint8_t *next = back + static_cast<std::ptrdiff_t>((y + 1) * _stride);

            unsigned left_column = upper[0] + current[0] + lower[0];
            unsigned center_column = upper[1] + current[1] + lower[1];
            unsigned right_column = upper[2] + current[2] + lower[2];
            unsigned window_sum = left_column + center_column + right_column;

#if defined(__SSE2__)
            static const __m128i kAliveVector = _mm_set1_epi8(1);
            static const __m128i kTwoVector = _mm_set1_epi8(2);
            static const __m128i kThreeVector = _mm_set1_epi8(3);

            int x = 1;
#if defined(__i386__) || defined(__x86_64__)
            if (_use_avx2) {
                x = step_row_avx2(upper, current, lower, next, _view_width);
            }
#endif
            for (; (x + kSimdWidth - 1) <= _view_width; x += kSimdWidth) {
                const __m128i upper_left = _mm_loadu_si128(reinterpret_cast<const __m128i *>(upper + x - 1));
                const __m128i upper_center = _mm_loadu_si128(reinterpret_cast<const __m128i *>(upper + x));
                const __m128i upper_right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(upper + x + 1));
                const __m128i current_left = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current + x - 1));
                const __m128i current_center = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current + x));
                const __m128i current_right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current + x + 1));
                const __m128i lower_left = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lower + x - 1));
                const __m128i lower_center = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lower + x));
                const __m128i lower_right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lower + x + 1));

                __m128i neighbors = _mm_add_epi8(upper_left, upper_center);
                neighbors = _mm_add_epi8(neighbors, upper_right);
                neighbors = _mm_add_epi8(neighbors, current_left);
                neighbors = _mm_add_epi8(neighbors, current_right);
                neighbors = _mm_add_epi8(neighbors, lower_left);
                neighbors = _mm_add_epi8(neighbors, lower_center);
                neighbors = _mm_add_epi8(neighbors, lower_right);

                const __m128i alive_mask = _mm_cmpeq_epi8(current_center, kAliveVector);
                const __m128i born_mask = _mm_cmpeq_epi8(neighbors, kThreeVector);
                const __m128i survive_mask = _mm_and_si128(alive_mask, _mm_cmpeq_epi8(neighbors, kTwoVector));
                const __m128i next_mask = _mm_or_si128(born_mask, survive_mask);
                const __m128i next_values = _mm_and_si128(next_mask, kAliveVector);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(next + x), next_values);
            }
            if (x <= _view_width) {
                left_column = upper[x - 1] + current[x - 1] + lower[x - 1];
                center_column = upper[x] + current[x] + lower[x];
                right_column = upper[x + 1] + current[x + 1] + lower[x + 1];
                window_sum = left_column + center_column + right_column;
            }
#else
            int x = 1;
#endif

            for (; x < _view_width; ++x) {
                const std::uint8_t was_alive = current[x];
                const unsigned neighbors = window_sum - was_alive;
                const std::uint8_t next_alive = kNextState[static_cast<std::size_t>((was_alive * 9U) + neighbors)];
                next[x] = next_alive;

                const unsigned next_column = upper[x + 2] + current[x + 2] + lower[x + 2];
                window_sum += next_column - left_column;
                left_column = center_column;
                center_column = right_column;
                right_column = next_column;
            }

            if (_view_width > 0) {
                const std::uint8_t was_alive = current[_view_width];
                const unsigned neighbors = window_sum - was_alive;
                const std::uint8_t next_alive = kNextState[static_cast<std::size_t>((was_alive * 9U) + neighbors)];
                next[_view_width] = next_alive;
            }

            if (render_target != nullptr) {
                auto *pixel_row = reinterpret_cast<std::uint32_t *>(
                        render_target->surface_bytes + static_cast<std::ptrdiff_t>(y) * render_target->pitch_bytes);
#if defined(__i386__) || defined(__x86_64__)
                if (render_target->use_avx2) {
                    paint_row_avx2(next + 1,
                                   pixel_row,
                                   _view_width,
                                   render_target->pixel_lut,
                                   render_target->stream_stores);
                } else {
                    paint_row_scalar(next + 1,
                                     pixel_row,
                                     _view_width,
                                     render_target->dead_pixel,
                                     render_target->alive_pixel);
                }
#else
                paint_row_scalar(next + 1,
                                 pixel_row,
                                 _view_width,
                                 render_target->dead_pixel,
                                 render_target->alive_pixel);
#endif
            }
        }

#if defined(__i386__) || defined(__x86_64__)
        if ((render_target != nullptr) && render_target->use_avx2 && render_target->stream_stores) {
            _mm_sfence();
        }
#endif
    }

    const int _view_width;
    const int _view_height;
    const int _stride;
    const int _parallelism;
    const bool _use_avx2;

    ThreadPool _thread_pool;
    bool _first_frame = true;

    std::vector<std::uint8_t> _front;
    std::vector<std::uint8_t> _back;
    std::vector<TaskRange> _task_ranges;
};

LifeBoard::LifeBoard(std::shared_ptr<CellSet> board, int threads, int width, int height)
        : _impl(std::make_unique<Impl>(std::move(board), threads, width, height)) {
}

LifeBoard::LifeBoard(CellBuffer cells, int threads, int width, int height)
        : _impl(std::make_unique<Impl>(std::move(cells), threads, width, height)) {
}

LifeBoard::~LifeBoard() = default;

LifeBoard::FrameView LifeBoard::iterate() {
    return _impl->iterate(nullptr);
}

LifeBoard::FrameView LifeBoard::iterate(const RenderTarget &render_target) {
    return _impl->iterate(&render_target);
}

LifeBoard::CellSet LifeBoard::snapshot() const {
    return _impl->snapshot();
}

bool LifeBoard::alive(Cell cell) const {
    return _impl->alive(cell);
}

int LifeBoard::width() const {
    return _impl->width();
}

int LifeBoard::height() const {
    return _impl->height();
}

int main(int argc, char **args) {
    (void) argc;
    (void) args;

    constexpr int kWidth = 3000;
    constexpr int kHeight = 1500;
    constexpr float kDensity = 0.05f;
    const int benchmark_frames = []() {
        const char *value = std::getenv("CLIFE_BENCH_FRAMES");
        return value != nullptr ? std::max(0, std::atoi(value)) : 0;
    }();

    const int worker_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    auto board = std::make_shared<LifeBoard>(make_seed_cells(kWidth, kHeight, kDensity),
                                             worker_threads,
                                             kWidth,
                                             kHeight);

#if defined(__i386__) || defined(__x86_64__)
    const bool use_render_avx2 = detect_avx2();
#else
    const bool use_render_avx2 = false;
#endif

    HeadlessFrameBuffer headless_buffer;
    X11FrameBuffer x11_buffer;
    std::uint8_t *surface_bytes = nullptr;
    int pitch_bytes = 0;
    PixelPalette palette{};
    bool stream_stores = false;

    if (benchmark_frames > 0) {
        headless_buffer = create_headless_frame_buffer(kWidth, kHeight);
        if (!headless_buffer.bytes) {
            std::cerr << "Failed to allocate benchmark frame buffer" << std::endl;
            return EXIT_FAILURE;
        }
        surface_bytes = headless_buffer.bytes.get();
        pitch_bytes = headless_buffer.pitch_bytes;
        palette = headless_buffer.palette;
        stream_stores = headless_buffer.stream_stores;
    } else {
        if (!x11_buffer.create(kWidth, kHeight)) {
            return EXIT_FAILURE;
        }
        surface_bytes = x11_buffer.pixels();
        pitch_bytes = x11_buffer.pitch_bytes();
        palette = x11_buffer.palette();
        stream_stores = x11_buffer.stream_stores();
    }

    const PixelLookupTable pixel_lut = make_pixel_lookup_table(palette.dead, palette.alive);
    const LifeBoard::RenderTarget render_target = {
            surface_bytes,
            pitch_bytes,
            palette.dead,
            palette.alive,
            pixel_lut.entries.data(),
            stream_stores,
            use_render_avx2,
    };

    const auto benchmark_start = std::chrono::steady_clock::now();
    bool running = true;
    int frames_rendered = 0;
    while (running && ((benchmark_frames == 0) || (frames_rendered < benchmark_frames))) {
        if (benchmark_frames == 0) {
            x11_buffer.poll_events(running);
            if (!running) {
                break;
            }
        }

        const LifeBoard::FrameView frame = board->iterate(render_target);

        if (benchmark_frames == 0) {
            x11_buffer.present(frame.view_width, frame.view_height);
        }
        ++frames_rendered;
    }

    if (benchmark_frames > 0) {
        const auto benchmark_end = std::chrono::steady_clock::now();
        const double seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(benchmark_end - benchmark_start).count();
        const double fps = seconds > 0.0 ? static_cast<double>(frames_rendered) / seconds : 0.0;
        std::cout << "benchmark_frames=" << frames_rendered
                  << " elapsed_seconds=" << seconds
                  << " fps=" << fps
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
