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
    using DirtySpan = LifeBoard::DirtySpan;

    constexpr std::uint32_t kDeadPixel = 0xFF000000;
    constexpr std::uint32_t kAlivePixel = 0xFFFFFFFF;
    constexpr std::array<std::uint32_t, 2> kPixelColors = {kDeadPixel, kAlivePixel};
    constexpr std::array<std::uint8_t, 18> kNextState = {
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
    };
#if defined(__i386__) || defined(__x86_64__)
    struct Avx2RowResult {
        int next_x;
        int min_changed;
        int max_changed;
    };

    [[nodiscard]] bool detect_avx2() {
#if defined(__GNUC__) || defined(__clang__)
        static const bool supported = __builtin_cpu_supports("avx2");
        return supported;
#else
        return false;
#endif
    }

    __attribute__((target("avx2"))) [[nodiscard]] Avx2RowResult step_row_avx2(
            const std::uint8_t *upper,
            const std::uint8_t *current,
            const std::uint8_t *lower,
            std::uint8_t *next,
            int view_width) {
        constexpr int kAvx2Width = 32;
        const __m256i alive_value = _mm256_set1_epi8(1);
        const __m256i two_value = _mm256_set1_epi8(2);
        const __m256i three_value = _mm256_set1_epi8(3);
        int min_changed = view_width;
        int max_changed = -1;
        int x = 1;

        for (; (x + kAvx2Width - 1) <= view_width; x += kAvx2Width) {
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

            const unsigned changed_bits =
                    static_cast<unsigned>(~_mm256_movemask_epi8(_mm256_cmpeq_epi8(current_center, next_values)));
            if (changed_bits != 0U) {
                if (max_changed < 0) {
                    min_changed = (x - 1) + static_cast<int>(__builtin_ctz(changed_bits));
                }
                max_changed = (x - 1) + (31 - static_cast<int>(__builtin_clz(changed_bits)));
            }
        }

        return {x, min_changed, max_changed};
    }
#endif
#if defined(__SSE2__)
    constexpr int kSimdWidth = 16;
    constexpr unsigned kSimdMask = 0xFFFFU;
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

    struct WorkerScratch {
        std::vector<DirtySpan> dirty_spans;
    };

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
              _task_ranges(static_cast<std::size_t>(_parallelism)),
              _scratch(static_cast<std::size_t>(_parallelism)) {
        initialize_scratch();

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
              _task_ranges(static_cast<std::size_t>(_parallelism)),
              _scratch(static_cast<std::size_t>(_parallelism)) {
        initialize_scratch();

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
            _dirty_spans.clear();
            return {&_front, &_dirty_spans, _view_width, _view_height, _stride, _stride + 1, true};
        }

        update_halo(_front);

        _thread_pool.parallel_for(static_cast<std::size_t>(_parallelism), [this](std::size_t task_index) {
            evaluate_chunk(task_index);
        });

        std::size_t total_spans = 0;
        for (int chunk = 0; chunk < _parallelism; ++chunk) {
            const WorkerScratch &scratch = _scratch[static_cast<std::size_t>(chunk)];
            total_spans += scratch.dirty_spans.size();
        }

        _dirty_spans.resize(total_spans);
        std::size_t offset = 0;
        for (int chunk = 0; chunk < _parallelism; ++chunk) {
            const WorkerScratch &scratch = _scratch[static_cast<std::size_t>(chunk)];
            std::copy(scratch.dirty_spans.begin(),
                      scratch.dirty_spans.end(),
                      _dirty_spans.begin() + static_cast<std::ptrdiff_t>(offset));
            offset += scratch.dirty_spans.size();
        }

        _front.swap(_back);

        return {&_front, &_dirty_spans, _view_width, _view_height, _stride, _stride + 1, false};
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
    void initialize_scratch() {
        const int rows_per_chunk = _view_height / _parallelism;
        const int extra_rows = _view_height % _parallelism;
        for (int task = 0; task < _parallelism; ++task) {
            const int begin_row = (task * rows_per_chunk) + std::min(task, extra_rows);
            const int row_count = rows_per_chunk + (task < extra_rows ? 1 : 0);
            _task_ranges[static_cast<std::size_t>(task)] = {begin_row, begin_row + row_count};
            _scratch[static_cast<std::size_t>(task)].dirty_spans.reserve(static_cast<std::size_t>(row_count));
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
        WorkerScratch &scratch = _scratch[task_index];
        scratch.dirty_spans.clear();
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

            int min_changed = _view_width;
            int max_changed = -1;
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
                const Avx2RowResult result = step_row_avx2(upper, current, lower, next, _view_width);
                x = result.next_x;
                if (result.max_changed >= 0) {
                    min_changed = result.min_changed;
                    max_changed = result.max_changed;
                }
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

                const unsigned changed_bits =
                        static_cast<unsigned>(~_mm_movemask_epi8(_mm_cmpeq_epi8(current_center, next_values))) & kSimdMask;
                if (changed_bits != 0U) {
                    if (max_changed < 0) {
                        min_changed = (x - 1) + static_cast<int>(__builtin_ctz(changed_bits));
                    }
                    max_changed = (x - 1) + (31 - static_cast<int>(__builtin_clz(changed_bits)));
                }
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

                if (next_alive != was_alive) {
                    if (max_changed < 0) {
                        min_changed = x - 1;
                    }
                    max_changed = x - 1;
                }

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

                if (next_alive != was_alive) {
                    if (max_changed < 0) {
                        min_changed = _view_width - 1;
                    }
                    max_changed = _view_width - 1;
                }
            }

            if (max_changed >= 0) {
                scratch.dirty_spans.push_back({
                        static_cast<std::uint16_t>(y),
                        static_cast<std::uint16_t>(min_changed),
                        static_cast<std::uint16_t>(max_changed),
                });
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
    std::vector<DirtySpan> _dirty_spans;
    std::vector<WorkerScratch> _scratch;
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

    const int worker_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    auto board = std::make_shared<LifeBoard>(make_seed_cells(kWidth, kHeight, kDensity),
                                             worker_threads,
                                             kWidth,
                                             kHeight);

    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return EXIT_FAILURE;
    }

    if (SDL_CreateWindowAndRenderer(kWidth, kHeight, 0, &window, &renderer) != 0) {
        std::cerr << "SDL_CreateWindowAndRenderer failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             kWidth,
                                             kHeight);
    if (texture == nullptr) {
        std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    std::vector<std::uint32_t> pixels(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), kDeadPixel);
    const int pitch_bytes = kWidth * static_cast<int>(sizeof(std::uint32_t));
    const std::size_t full_upload_pixel_threshold = pixels.size() / 3;
    const std::size_t full_upload_span_threshold = pixels.size() / static_cast<std::size_t>(kWidth * 8);

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event) != 0) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if ((event.type == SDL_WINDOWEVENT) && (event.window.event == SDL_WINDOWEVENT_CLOSE)) {
                running = false;
            }
        }

        const LifeBoard::FrameView frame = board->iterate();
        if (frame.full_refresh) {
            const std::uint8_t *cells = frame.cells->data();
            for (int row = 0; row < frame.view_height; ++row) {
                const int src_index = frame.top_left_index + row * frame.stride;
                const std::uint8_t *cell_row = cells + src_index;
                std::uint32_t *pixel_row = pixels.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(frame.view_width);
                for (int column = 0; column < frame.view_width; ++column) {
                    pixel_row[column] = kPixelColors[cell_row[column]];
                }
            }
            SDL_UpdateTexture(texture, nullptr, pixels.data(), pitch_bytes);
        } else if (!frame.dirty_spans->empty()) {
            const std::uint8_t *cells = frame.cells->data();
            std::size_t dirty_pixels = 0;
            int min_dirty_row = frame.view_height;
            int max_dirty_row = -1;
            int min_dirty_x = frame.view_width;
            int max_dirty_x = -1;
            for (const DirtySpan &span: *frame.dirty_spans) {
                const int row = static_cast<int>(span.row);
                const int min_x = static_cast<int>(span.min_x);
                const int max_x = static_cast<int>(span.max_x);
                const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(frame.view_width);
                const std::size_t src_index =
                        static_cast<std::size_t>(frame.top_left_index + row * frame.stride);
                dirty_pixels += static_cast<std::size_t>(max_x - min_x + 1);
                min_dirty_row = std::min(min_dirty_row, row);
                max_dirty_row = std::max(max_dirty_row, row);
                min_dirty_x = std::min(min_dirty_x, min_x);
                max_dirty_x = std::max(max_dirty_x, max_x);
                const std::uint8_t *cell_row = cells + src_index + static_cast<std::size_t>(min_x);
                std::uint32_t *pixel_row = pixels.data() + row_offset + static_cast<std::size_t>(min_x);
                for (int x = min_x; x <= max_x; ++x) {
                    *pixel_row++ = kPixelColors[*cell_row++];
                }
            }

            const std::size_t dirty_box_area =
                    static_cast<std::size_t>(max_dirty_row - min_dirty_row + 1) *
                    static_cast<std::size_t>(max_dirty_x - min_dirty_x + 1);
            if ((dirty_pixels >= full_upload_pixel_threshold) ||
                (frame.dirty_spans->size() >= full_upload_span_threshold)) {
                SDL_UpdateTexture(texture, nullptr, pixels.data(), pitch_bytes);
            } else if (dirty_box_area <= (dirty_pixels * 2U)) {
                const SDL_Rect rect = {min_dirty_x, min_dirty_row, max_dirty_x - min_dirty_x + 1, max_dirty_row - min_dirty_row + 1};
                SDL_UpdateTexture(texture,
                                  &rect,
                                  pixels.data() + static_cast<std::size_t>(min_dirty_row * frame.view_width + min_dirty_x),
                                  pitch_bytes);
            } else {
                for (const DirtySpan &span: *frame.dirty_spans) {
                    const int row = static_cast<int>(span.row);
                    const int min_x = static_cast<int>(span.min_x);
                    const int max_x = static_cast<int>(span.max_x);
                    const std::size_t row_offset =
                            static_cast<std::size_t>(row) * static_cast<std::size_t>(frame.view_width);
                    const SDL_Rect rect = {min_x, row, max_x - min_x + 1, 1};
                    SDL_UpdateTexture(texture,
                                      &rect,
                                      pixels.data() + row_offset + static_cast<std::size_t>(min_x),
                                      rect.w * static_cast<int>(sizeof(std::uint32_t)));
                }
            }
        }

        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
