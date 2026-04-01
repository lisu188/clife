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


namespace {
    using Cell = LifeBoard::Cell;
    using CellSet = LifeBoard::CellSet;
    using CellBuffer = LifeBoard::CellBuffer;

    constexpr std::uint32_t kDeadPixel = 0xFF000000;
    constexpr std::uint32_t kAlivePixel = 0xFFFFFFFF;
    constexpr std::array<std::uint32_t, 2> kPixelColors = {kDeadPixel, kAlivePixel};
    constexpr std::array<std::uint8_t, 18> kNextState = {
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
    };
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

    __attribute__((target("avx2"))) void paint_frame_avx2(
            const std::uint8_t *cells,
            std::uint8_t *surface_bytes,
            int view_width,
            int view_height,
            int stride,
            int top_left_index,
            int pitch_bytes) {
        constexpr int kRenderBlockWidth = 32;
        const __m256i zero = _mm256_setzero_si256();
        const __m256i dead_pixels = _mm256_set1_epi32(static_cast<int>(kDeadPixel));
        const __m256i alive_pixels = _mm256_set1_epi32(static_cast<int>(kAlivePixel));

        for (int row = 0; row < view_height; ++row) {
            const std::uint8_t *cell_row = cells + top_left_index + row * stride;
            auto *pixel_row =
                    reinterpret_cast<std::uint32_t *>(surface_bytes + static_cast<std::ptrdiff_t>(row) * pitch_bytes);

            int x = 0;
            for (; (x + kRenderBlockWidth) <= view_width; x += kRenderBlockWidth) {
                for (int block = 0; block < kRenderBlockWidth; block += 8) {
                    const __m128i source = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(cell_row + x + block));
                    const __m256i expanded = _mm256_cvtepu8_epi32(source);
                    const __m256i alive_mask = _mm256_cmpgt_epi32(expanded, zero);
                    const __m256i colored = _mm256_blendv_epi8(dead_pixels, alive_pixels, alive_mask);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(pixel_row + x + block), colored);
                }
            }

            for (; x < view_width; ++x) {
                pixel_row[x] = kPixelColors[cell_row[x]];
            }
        }
    }
#endif

    void paint_frame_scalar(
            const std::uint8_t *cells,
            std::uint8_t *surface_bytes,
            int view_width,
            int view_height,
            int stride,
            int top_left_index,
            int pitch_bytes) {
        for (int row = 0; row < view_height; ++row) {
            const std::uint8_t *cell_row = cells + top_left_index + row * stride;
            auto *pixel_row =
                    reinterpret_cast<std::uint32_t *>(surface_bytes + static_cast<std::ptrdiff_t>(row) * pitch_bytes);
            for (int x = 0; x < view_width; ++x) {
                pixel_row[x] = kPixelColors[cell_row[x]];
            }
        }
    }

    void paint_frame(
            const std::uint8_t *cells,
            std::uint8_t *surface_bytes,
            int view_width,
            int view_height,
            int stride,
            int top_left_index,
            int pitch_bytes,
            bool use_avx2) {
#if defined(__i386__) || defined(__x86_64__)
        if (use_avx2) {
            paint_frame_avx2(cells, surface_bytes, view_width, view_height, stride, top_left_index, pitch_bytes);
            return;
        }
#endif
        paint_frame_scalar(cells, surface_bytes, view_width, view_height, stride, top_left_index, pitch_bytes);
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

    [[nodiscard]] LifeBoard::FrameView iterate() {
        if (_first_frame) {
            _first_frame = false;
            return {&_front, _view_width, _view_height, _stride, _stride + 1};
        }

        update_halo(_front);

        _thread_pool.parallel_for(static_cast<std::size_t>(_parallelism), [this](std::size_t task_index) {
            evaluate_chunk(task_index);
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

    void evaluate_chunk(std::size_t task_index) {
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
        }
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
    return _impl->iterate();
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

    SDL_Window *window = nullptr;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return EXIT_FAILURE;
    }

    window = SDL_CreateWindow("clife", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, kWidth, kHeight, 0);
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Surface *window_surface = SDL_GetWindowSurface(window);
    if ((window_surface == nullptr) || (window_surface->format->BytesPerPixel != 4)) {
        std::cerr << "SDL_GetWindowSurface failed or returned an unsupported format" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

#if defined(__i386__) || defined(__x86_64__)
    const bool use_render_avx2 = detect_avx2();
#else
    const bool use_render_avx2 = false;
#endif

    const auto benchmark_start = std::chrono::steady_clock::now();
    bool running = true;
    int frames_rendered = 0;
    while (running && ((benchmark_frames == 0) || (frames_rendered < benchmark_frames))) {
        if (benchmark_frames == 0) {
            SDL_Event event;
            while (SDL_PollEvent(&event) != 0) {
                if (event.type == SDL_QUIT) {
                    running = false;
                }
                if ((event.type == SDL_WINDOWEVENT) && (event.window.event == SDL_WINDOWEVENT_CLOSE)) {
                    running = false;
                }
            }
            if (!running) {
                break;
            }
        }

        const LifeBoard::FrameView frame = board->iterate();
        if (SDL_MUSTLOCK(window_surface) && (SDL_LockSurface(window_surface) != 0)) {
            std::cerr << "SDL_LockSurface failed: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return EXIT_FAILURE;
        }

        const std::uint8_t *cells = frame.cells->data();
        auto *surface_bytes = static_cast<std::uint8_t *>(window_surface->pixels);
        paint_frame(cells,
                    surface_bytes,
                    frame.view_width,
                    frame.view_height,
                    frame.stride,
                    frame.top_left_index,
                    window_surface->pitch,
                    use_render_avx2);

        if (SDL_MUSTLOCK(window_surface)) {
            SDL_UnlockSurface(window_surface);
        }
        if (SDL_UpdateWindowSurface(window) != 0) {
            std::cerr << "SDL_UpdateWindowSurface failed: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return EXIT_FAILURE;
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

    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
