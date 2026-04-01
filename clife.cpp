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
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>


namespace {
    using Cell = LifeBoard::Cell;
    using CellSet = LifeBoard::CellSet;
    using IndexList = LifeBoard::IndexList;

    constexpr std::uint32_t kDeadPixel = 0xFF000000;
    constexpr std::uint32_t kAlivePixel = 0xFFFFFFFF;
    constexpr int kSimulationPadding = 256;

    bool get_cell(const CellSet &board, const Cell &coords) {
        return board.find(coords) != board.end();
    }

    bool set_cell(CellSet &board, const Cell &coords, bool value) {
        const CellSet::iterator it = board.find(coords);
        if (value && (it == board.end())) {
            board.insert(coords);
            return true;
        }
        if (!value && (it != board.end())) {
            board.erase(coords);
            return true;
        }
        return false;
    }

    bool flip_cell(CellSet &board, const Cell &cell) {
        return set_cell(board, cell, !get_cell(board, cell));
    }

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

    struct Bounds {
        int min_x = 0;
        int max_x = -1;
        int min_y = 0;
        int max_y = -1;

        [[nodiscard]] bool empty() const {
            return (max_x < min_x) || (max_y < min_y);
        }

        [[nodiscard]] int height() const {
            return empty() ? 0 : (max_y - min_y + 1);
        }

        [[nodiscard]] int width() const {
            return empty() ? 0 : (max_x - min_x + 1);
        }

        [[nodiscard]] std::size_t area() const {
            return static_cast<std::size_t>(width()) * static_cast<std::size_t>(height());
        }

        void reset() {
            min_x = 0;
            max_x = -1;
            min_y = 0;
            max_y = -1;
        }

        void include(int x, int y) {
            if (empty()) {
                min_x = max_x = x;
                min_y = max_y = y;
                return;
            }
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }

        void merge(const Bounds &other) {
            if (other.empty()) {
                return;
            }
            if (empty()) {
                *this = other;
                return;
            }
            min_x = std::min(min_x, other.min_x);
            max_x = std::max(max_x, other.max_x);
            min_y = std::min(min_y, other.min_y);
            max_y = std::max(max_y, other.max_y);
        }

        [[nodiscard]] Bounds expanded(int margin, int width, int height) const {
            if (empty()) {
                return {};
            }

            Bounds expanded_bounds;
            expanded_bounds.min_x = std::max(0, min_x - margin);
            expanded_bounds.max_x = std::min(width - 1, max_x + margin);
            expanded_bounds.min_y = std::max(0, min_y - margin);
            expanded_bounds.max_y = std::min(height - 1, max_y + margin);
            return expanded_bounds;
        }
    };

    struct WorkerScratch {
        IndexList changed;
        std::size_t changed_count = 0;
        Bounds live_bounds;
        int live_count = 0;
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

        [[nodiscard]] std::size_t parallelism() const {
            return _threads.size() + 1;
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
              _origin_x(kSimulationPadding),
              _origin_y(kSimulationPadding),
              _sim_width(_view_width + (2 * kSimulationPadding)),
              _sim_height(_view_height + (2 * kSimulationPadding)),
              _stride(_sim_width + 2),
              _parallelism(std::max(1, std::min(threads, _sim_height))),
              _thread_pool(static_cast<std::size_t>(std::max(0, _parallelism - 1))),
              _front(static_cast<std::size_t>(_stride) * static_cast<std::size_t>(_sim_height + 2), 0),
              _back(_front.size(), 0),
              _neighbor_counts(_front.size(), 0),
              _scratch(static_cast<std::size_t>(_parallelism)) {
        _neighbor_offsets = {{
                -_stride - 1, -_stride, -_stride + 1,
                -1,           1,
                _stride - 1,  _stride,  _stride + 1,
        }};

        if (!board) {
            return;
        }

        for (const Cell &cell: *board) {
            if (!in_bounds(cell)) {
                continue;
            }

            const int index = to_index(cell);
            if (_front[static_cast<std::size_t>(index)] != 0) {
                continue;
            }

            _front[static_cast<std::size_t>(index)] = 1;
            ++_live_count;
            _active_bounds.include(cell.first + _origin_x, cell.second + _origin_y);
        }

        const int max_chunk_rows = (_sim_height + _parallelism - 1) / _parallelism;
        const std::size_t scratch_capacity =
                static_cast<std::size_t>(max_chunk_rows) * static_cast<std::size_t>(_sim_width);
        for (WorkerScratch &scratch: _scratch) {
            scratch.changed.resize(scratch_capacity);
        }
        _changed_indices.reserve(static_cast<std::size_t>(_sim_width) * static_cast<std::size_t>(_sim_height));

        build_neighbor_counts();
    }

    [[nodiscard]] LifeBoard::FrameView iterate() {
        if (_first_frame) {
            _first_frame = false;
            _changed_indices.clear();
            return {&_front, &_changed_indices, _view_width, _view_height, _stride, _origin_x, _origin_y, true};
        }

        _changed_indices.clear();
        if (_active_bounds.empty()) {
            return {&_front, &_changed_indices, _view_width, _view_height, _stride, _origin_x, _origin_y, false};
        }

        const Bounds eval_bounds = _active_bounds.expanded(1, _sim_width, _sim_height);
        clear_back_bounds(_back_live_bounds);

        const int chunk_count = std::min(_parallelism, eval_bounds.height());
        _thread_pool.parallel_for(static_cast<std::size_t>(chunk_count), [this, eval_bounds, chunk_count](std::size_t task_index) {
            evaluate_chunk(task_index, eval_bounds, chunk_count);
        });

        Bounds next_bounds;
        int next_live_count = 0;
        std::size_t total_changes = 0;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const WorkerScratch &scratch = _scratch[static_cast<std::size_t>(chunk)];
            total_changes += scratch.changed_count;
            next_live_count += scratch.live_count;
            next_bounds.merge(scratch.live_bounds);
        }

        _changed_indices.resize(total_changes);
        std::size_t offset = 0;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const WorkerScratch &scratch = _scratch[static_cast<std::size_t>(chunk)];
            std::copy_n(scratch.changed.begin(),
                        static_cast<std::ptrdiff_t>(scratch.changed_count),
                        _changed_indices.begin() + static_cast<std::ptrdiff_t>(offset));
            offset += scratch.changed_count;
        }

        _front.swap(_back);
        _back_live_bounds = _active_bounds;
        _live_count = next_live_count;
        _active_bounds = next_bounds;
        apply_neighbor_deltas();

        return {&_front, &_changed_indices, _view_width, _view_height, _stride, _origin_x, _origin_y, false};
    }

    [[nodiscard]] CellSet snapshot() const {
        CellSet board;
        board.reserve(static_cast<std::size_t>(_live_count));
        if (_active_bounds.empty()) {
            return board;
        }

        for (int y = _active_bounds.min_y; y <= _active_bounds.max_y; ++y) {
            int index = (y + 1) * _stride + (_active_bounds.min_x + 1);
            for (int x = _active_bounds.min_x; x <= _active_bounds.max_x; ++x, ++index) {
                if (_front[static_cast<std::size_t>(index)] != 0) {
                    board.emplace(x - _origin_x, y - _origin_y);
                }
            }
        }
        return board;
    }

    [[nodiscard]] bool alive(Cell cell) const {
        return in_bounds(cell) && (_front[static_cast<std::size_t>(to_index(cell))] != 0);
    }

    [[nodiscard]] int width() const {
        return _view_width;
    }

    [[nodiscard]] int height() const {
        return _view_height;
    }

private:
    [[nodiscard]] bool in_bounds(Cell cell) const {
        return (cell.first >= -_origin_x) && (cell.first < (_sim_width - _origin_x)) &&
               (cell.second >= -_origin_y) && (cell.second < (_sim_height - _origin_y));
    }

    [[nodiscard]] int to_index(Cell cell) const {
        return (cell.second + _origin_y + 1) * _stride + (cell.first + _origin_x + 1);
    }

    void build_neighbor_counts() {
        std::fill(_neighbor_counts.begin(), _neighbor_counts.end(), static_cast<std::uint8_t>(0));
        if (_active_bounds.empty()) {
            return;
        }

        for (int y = _active_bounds.min_y; y <= _active_bounds.max_y; ++y) {
            int index = (y + 1) * _stride + (_active_bounds.min_x + 1);
            for (int x = _active_bounds.min_x; x <= _active_bounds.max_x; ++x, ++index) {
                if (_front[static_cast<std::size_t>(index)] == 0) {
                    continue;
                }
                for (const int offset: _neighbor_offsets) {
                    ++_neighbor_counts[static_cast<std::size_t>(index + offset)];
                }
            }
        }
    }

    void clear_back_bounds(const Bounds &bounds) {
        if (bounds.empty()) {
            return;
        }

        const int clear_width = bounds.max_x - bounds.min_x + 1;
        for (int y = bounds.min_y; y <= bounds.max_y; ++y) {
            const int row_start = (y + 1) * _stride + (bounds.min_x + 1);
            std::fill_n(_back.begin() + row_start, clear_width, static_cast<std::uint8_t>(0));
        }
    }

    void apply_neighbor_deltas() {
        for (const int index: _changed_indices) {
            const int delta = (_front[static_cast<std::size_t>(index)] != 0) ? 1 : -1;
            for (const int offset: _neighbor_offsets) {
                const std::size_t neighbor = static_cast<std::size_t>(index + offset);
                _neighbor_counts[neighbor] = static_cast<std::uint8_t>(
                        static_cast<int>(_neighbor_counts[neighbor]) + delta);
            }
        }
    }

    void evaluate_chunk(std::size_t task_index, Bounds eval_bounds, int chunk_count) {
        WorkerScratch &scratch = _scratch[task_index];
        scratch.changed_count = 0;
        scratch.live_bounds.reset();
        scratch.live_count = 0;

        const int task = static_cast<int>(task_index);
        const int rows_per_chunk = eval_bounds.height() / chunk_count;
        const int extra_rows = eval_bounds.height() % chunk_count;
        const int begin_row = eval_bounds.min_y + (task * rows_per_chunk) + std::min(task, extra_rows);
        const int row_count = rows_per_chunk + (task < extra_rows ? 1 : 0);
        if (row_count <= 0) {
            return;
        }

        const int end_row = begin_row + row_count;
        const int min_x = eval_bounds.min_x;
        const int max_x = eval_bounds.max_x;

        for (int y = begin_row; y < end_row; ++y) {
            int index = (y + 1) * _stride + (min_x + 1);
            for (int x = min_x; x <= max_x; ++x, ++index) {
                const std::uint8_t was_alive = _front[static_cast<std::size_t>(index)];
                const std::uint8_t neighbors = _neighbor_counts[static_cast<std::size_t>(index)];
                const std::uint8_t next_alive = static_cast<std::uint8_t>(
                        (neighbors == 3) | ((neighbors == 2) & was_alive));
                _back[static_cast<std::size_t>(index)] = next_alive;

                if (next_alive != was_alive) {
                    scratch.changed[scratch.changed_count++] = index;
                }
                if (next_alive != 0) {
                    ++scratch.live_count;
                    scratch.live_bounds.include(x, y);
                }
            }
        }
    }

    const int _view_width;
    const int _view_height;
    const int _origin_x;
    const int _origin_y;
    const int _sim_width;
    const int _sim_height;
    const int _stride;
    const int _parallelism;

    ThreadPool _thread_pool;
    int _live_count = 0;
    bool _first_frame = true;
    Bounds _active_bounds;
    Bounds _back_live_bounds;

    std::vector<std::uint8_t> _front;
    std::vector<std::uint8_t> _back;
    std::vector<std::uint8_t> _neighbor_counts;
    std::array<int, 8> _neighbor_offsets = {};
    IndexList _changed_indices;
    std::vector<WorkerScratch> _scratch;
};

LifeBoard::LifeBoard(std::shared_ptr<CellSet> board, int threads, int width, int height)
        : _impl(std::make_unique<Impl>(std::move(board), threads, width, height)) {
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
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    constexpr int kWidth = 500;
    constexpr int kHeight = 500;
    constexpr float kDensity = 0.8f;

    const int seeds = static_cast<int>(kWidth * kHeight * kDensity);
    auto seed_board = std::make_shared<CellSet>();
    seed_board->reserve(static_cast<std::size_t>(seeds));
    for (int index = 0; index < seeds; ++index) {
        flip_cell(*seed_board, {std::rand() % kWidth, std::rand() % kHeight});
    }

    const int worker_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    auto board = std::make_shared<LifeBoard>(seed_board, worker_threads, kWidth, kHeight);

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
    std::vector<int> dirty_rows;
    std::vector<int> dirty_min_x(static_cast<std::size_t>(kHeight), kWidth);
    std::vector<int> dirty_max_x(static_cast<std::size_t>(kHeight), -1);

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
            for (int row = 0; row < frame.view_height; ++row) {
                const int src_index = (row + frame.origin_y + 1) * frame.stride + (frame.origin_x + 1);
                const int dst_index = row * frame.view_width;
                for (int column = 0; column < frame.view_width; ++column) {
                    pixels[static_cast<std::size_t>(dst_index + column)] =
                            (*frame.cells)[static_cast<std::size_t>(src_index + column)] != 0
                            ? kAlivePixel
                            : kDeadPixel;
                }
            }
            SDL_UpdateTexture(texture, nullptr, pixels.data(), frame.view_width * static_cast<int>(sizeof(std::uint32_t)));
        } else if (!frame.changed->empty()) {
            dirty_rows.clear();
            for (const int index: *frame.changed) {
                const int x = index % frame.stride - 1 - frame.origin_x;
                const int y = index / frame.stride - 1 - frame.origin_y;
                if ((x < 0) || (x >= frame.view_width) || (y < 0) || (y >= frame.view_height)) {
                    continue;
                }

                const std::size_t pixel_index = static_cast<std::size_t>(y * frame.view_width + x);
                pixels[pixel_index] = (*frame.cells)[static_cast<std::size_t>(index)] != 0 ? kAlivePixel : kDeadPixel;

                if (dirty_max_x[static_cast<std::size_t>(y)] < 0) {
                    dirty_rows.push_back(y);
                    dirty_min_x[static_cast<std::size_t>(y)] = x;
                    dirty_max_x[static_cast<std::size_t>(y)] = x;
                } else {
                    dirty_min_x[static_cast<std::size_t>(y)] = std::min(dirty_min_x[static_cast<std::size_t>(y)], x);
                    dirty_max_x[static_cast<std::size_t>(y)] = std::max(dirty_max_x[static_cast<std::size_t>(y)], x);
                }
            }

            for (const int row: dirty_rows) {
                const int min_x = dirty_min_x[static_cast<std::size_t>(row)];
                const int max_x = dirty_max_x[static_cast<std::size_t>(row)];
                const SDL_Rect rect = {min_x, row, max_x - min_x + 1, 1};
                SDL_UpdateTexture(texture,
                                  &rect,
                                  pixels.data() + static_cast<std::size_t>(row * frame.view_width + min_x),
                                  rect.w * static_cast<int>(sizeof(std::uint32_t)));
                dirty_min_x[static_cast<std::size_t>(row)] = frame.view_width;
                dirty_max_x[static_cast<std::size_t>(row)] = -1;
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
