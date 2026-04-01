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
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>


namespace vstd {
    std::function<void(std::function<void()>)> get_call_later_handler() {
        return [](std::function<void()> f) {
            vstd::event_loop<>::instance()->invoke(f);
        };
    }

    std::function<void(std::function<void()>)> get_call_async_handler() {
        return [](std::function<void()> f) {
            static std::shared_ptr<vstd::thread_pool<16>> pool = std::make_shared<vstd::thread_pool<16>>()->start();
            pool->execute(f);
        };
    }

    std::function<void(std::function<void()>)> get_call_later_block_handler() {
        return [](std::function<void()> f) {
            vstd::event_loop<>::instance()->await(f);
        };
    }

    std::function<void(std::function<bool()>)> get_wait_until_handler() {
        return [](std::function<bool()> pred) {
            vstd::call_later_block([pred]() {
                while (!pred()) {
                    vstd::event_loop<>::instance()->run();
                }
            });
        };
    }

    std::function<void(int, std::function<void()>)> get_call_delayed_later_handler() {
        return [](int t, std::function<void()> f) {
            vstd::event_loop<>::instance()->delay(t, f);
        };
    }

    std::function<void(int, std::function<void()>)> get_call_delayed_async_handler() {
        return [](int t, std::function<void()> f) {
            vstd::event_loop<>::instance()->delay(t, [f]() {
                vstd::async(f);
            });
        };
    }
}

namespace {
    using Cell = LifeBoard::Cell;
    using CellSet = LifeBoard::CellSet;
    constexpr std::uint32_t kDeadPixel = 0xFF000000;
    constexpr std::uint32_t kAlivePixel = 0xFFFFFFFF;
    constexpr int kSimulationPadding = 256;

    bool get_cell(const CellSet &board, const Cell &coords) {
        return board.find(coords) != board.end();
    }

    bool set_cell(CellSet &board, const Cell &coords, bool val) {
        const CellSet::iterator it = board.find(coords);
        if (val && (it == board.end())) {
            board.insert(coords);
            return true;
        } else if (!val && (it != board.end())) {
            board.erase(coords);
            return true;
        }
        return false;
    }

    bool flip_cell(CellSet &board, const Cell &cell) {
        return set_cell(board, cell, !get_cell(board, cell));
    }
}

LifeBoard::LifeBoard(std::shared_ptr<CellSet> board, int threads, int width, int height)
        : _width(std::max(1, width) + (2 * kSimulationPadding)),
          _height(std::max(1, height) + (2 * kSimulationPadding)),
          _stride(_width + 2),
          _view_width(std::max(1, width)),
          _view_height(std::max(1, height)),
          _origin_x(kSimulationPadding),
          _origin_y(kSimulationPadding),
          _participants(std::max(1, std::min(threads, _height))) {
    if ((width <= 0) || (height <= 0)) {
        _view_width = 1;
        _view_height = 1;
        if (board) {
            for (const Cell &cell: *board) {
                if ((cell.first >= 0) && (cell.second >= 0)) {
                    _view_width = std::max(_view_width, cell.first + 1);
                    _view_height = std::max(_view_height, cell.second + 1);
                }
            }
        }
        _width = _view_width + (2 * kSimulationPadding);
        _height = _view_height + (2 * kSimulationPadding);
        _stride = _width + 2;
        _participants = std::max(1, std::min(threads, _height));
    }

    const std::size_t grid_size = static_cast<std::size_t>(_stride) * static_cast<std::size_t>(_height + 2);
    _front.assign(grid_size, 0);
    _back.assign(grid_size, 0);
    _neighbor_counts.assign(grid_size, 0);
    _neighbor_offsets = {{
            -_stride - 1, -_stride, -_stride + 1,
            -1,           1,
            _stride - 1,  _stride,  _stride + 1,
    }};

    if (board) {
        for (const Cell &cell: *board) {
            if (!in_bounds(cell)) {
                continue;
            }
            const int index = to_index(cell);
            if (_front[index] != 0) {
                continue;
            }
            _front[index] = 1;
            _live_cells++;
        }
    }

    build_neighbor_counts();

    _chunk_ranges.resize(static_cast<std::size_t>(_participants));
    _chunk_changes.resize(static_cast<std::size_t>(_participants));
    int next_row = 0;
    for (int chunk = 0; chunk < _participants; ++chunk) {
        const int rows = (_height / _participants) + (chunk < (_height % _participants) ? 1 : 0);
        _chunk_ranges[static_cast<std::size_t>(chunk)] = {next_row, next_row + rows};
        next_row += rows;
    }

    _workers.reserve(static_cast<std::size_t>(std::max(0, _participants - 1)));
    for (int chunk = 1; chunk < _participants; ++chunk) {
        _workers.emplace_back(&LifeBoard::worker_loop, this, static_cast<std::size_t>(chunk));
    }
}

LifeBoard::~LifeBoard() {
    if (_workers.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(_worker_mutex);
        _stop_workers = true;
    }
    _worker_cv.notify_all();

    for (std::thread &worker: _workers) {
        worker.join();
    }
}

LifeBoard::FrameView LifeBoard::iterate() {
    if (_first_frame) {
        _first_frame = false;
        _changed_indices.clear();
        return {&_front, &_changed_indices, _view_width, _view_height, _width, _height, _stride, _origin_x, _origin_y, true};
    }

    if (_participants > 1) {
        {
            std::lock_guard<std::mutex> lock(_worker_mutex);
            _completed_workers = 0;
            ++_job_generation;
        }
        _worker_cv.notify_all();
    }

    evaluate_chunk(0);

    if (_participants > 1) {
        std::unique_lock<std::mutex> lock(_worker_mutex);
        _done_cv.wait(lock, [this]() {
            return _completed_workers == _workers.size();
        });
    }

    std::size_t total_changes = 0;
    for (const IndexList &chunk: _chunk_changes) {
        total_changes += chunk.size();
    }

    _changed_indices.clear();
    _changed_indices.reserve(total_changes);
    for (const IndexList &chunk: _chunk_changes) {
        _changed_indices.insert(_changed_indices.end(), chunk.begin(), chunk.end());
    }

    _front.swap(_back);
    for (const int index: _changed_indices) {
        _live_cells += (_front[index] != 0) ? 1 : -1;
    }
    apply_neighbor_deltas();

    return {&_front, &_changed_indices, _view_width, _view_height, _width, _height, _stride, _origin_x, _origin_y, false};
}

LifeBoard::CellSet LifeBoard::snapshot() const {
    CellSet board;
    board.reserve(static_cast<std::size_t>(_live_cells));
    for (int y = 0; y < _height; ++y) {
        int index = (y + 1) * _stride + 1;
        for (int x = 0; x < _width; ++x, ++index) {
            if (_front[index] != 0) {
                board.emplace(x - _origin_x, y - _origin_y);
            }
        }
    }
    return board;
}

bool LifeBoard::alive(Cell cell) const {
    return in_bounds(cell) && (_front[to_index(cell)] != 0);
}

int LifeBoard::width() const {
    return _view_width;
}

int LifeBoard::height() const {
    return _view_height;
}

void LifeBoard::evaluate_chunk(std::size_t chunk_index) {
    IndexList &changed = _chunk_changes[chunk_index];
    changed.clear();

    const RowRange range = _chunk_ranges[chunk_index];
    if (range.begin >= range.end) {
        return;
    }

    changed.reserve(static_cast<std::size_t>(range.end - range.begin) * static_cast<std::size_t>(_width) / 3);

    const std::vector<std::uint8_t> &front = _front;
    const std::vector<std::uint8_t> &neighbor_counts = _neighbor_counts;
    std::vector<std::uint8_t> &back = _back;
    for (int row = range.begin; row < range.end; ++row) {
        int index = (row + 1) * _stride + 1;
        for (int column = 0; column < _width; ++column, ++index) {
            const bool was_alive = front[index] != 0;
            const std::uint8_t neighbors = neighbor_counts[index];
            const std::uint8_t next_state = ((neighbors == 3) || (was_alive && (neighbors == 2))) ? 1 : 0;
            back[index] = next_state;
            if (next_state != front[index]) {
                changed.push_back(index);
            }
        }
    }
}

void LifeBoard::worker_loop(std::size_t chunk_index) {
    std::size_t seen_generation = 0;
    std::unique_lock<std::mutex> lock(_worker_mutex);
    while (true) {
        _worker_cv.wait(lock, [this, &seen_generation]() {
            return _stop_workers || (_job_generation != seen_generation);
        });

        if (_stop_workers) {
            return;
        }

        seen_generation = _job_generation;
        lock.unlock();
        evaluate_chunk(chunk_index);
        lock.lock();

        ++_completed_workers;
        if (_completed_workers == _workers.size()) {
            _done_cv.notify_one();
        }
    }
}

void LifeBoard::build_neighbor_counts() {
    std::fill(_neighbor_counts.begin(), _neighbor_counts.end(), static_cast<std::uint8_t>(0));
    for (int row = 0; row < _height; ++row) {
        int index = (row + 1) * _stride + 1;
        for (int column = 0; column < _width; ++column, ++index) {
            if (_front[index] == 0) {
                continue;
            }

            for (const int offset: _neighbor_offsets) {
                ++_neighbor_counts[index + offset];
            }
        }
    }
}

void LifeBoard::apply_neighbor_deltas() {
    for (const int index: _changed_indices) {
        const int delta = (_front[index] != 0) ? 1 : -1;
        for (const int offset: _neighbor_offsets) {
            const int neighbor = index + offset;
            _neighbor_counts[neighbor] = static_cast<std::uint8_t>(
                    static_cast<int>(_neighbor_counts[neighbor]) + delta);
        }
    }
}

bool LifeBoard::in_bounds(Cell cell) const {
    return (cell.first >= -_origin_x) && (cell.first < (_width - _origin_x)) &&
           (cell.second >= -_origin_y) && (cell.second < (_height - _origin_y));
}

int LifeBoard::to_index(Cell cell) const {
    return (cell.second + _origin_y + 1) * _stride + (cell.first + _origin_x + 1);
}

Cell LifeBoard::to_cell(int index) const {
    return {index % _stride - 1 - _origin_x, index / _stride - 1 - _origin_y};
}

int main(int argc, char **args) {
    srand(time(0));
    int SIZEX = 500;
    int SIZEY = 500;
    float scale = 1;
    float factor = 0.8;
    int seeds = (int) (SIZEX * SIZEY * factor);
    std::shared_ptr<CellSet> tmp = std::make_shared<CellSet>();
    tmp->reserve(seeds);
    for (int i = 0; i < seeds; i++) {
        flip_cell(*tmp, Cell(rand() % SIZEX, rand() % SIZEY));
    }
    const int worker_threads = std::max(1u, std::thread::hardware_concurrency());
    std::shared_ptr<LifeBoard> board = std::make_shared<LifeBoard>(tmp, worker_threads, SIZEX, SIZEY);

    SDL_Window *window = 0;
    SDL_Renderer *renderer = 0;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return EXIT_FAILURE;
    }

    if (SDL_CreateWindowAndRenderer(SIZEX * scale, SIZEY * scale, 0, &window, &renderer) != 0) {
        std::cerr << "SDL_CreateWindowAndRenderer failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             SIZEX,
                                             SIZEY);
    if (texture == nullptr) {
        std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    auto loop = vstd::event_loop<>::instance();
    loop->registerFrameCallback(
            [board,
             renderer,
             texture,
             pixels = std::vector<std::uint32_t>(static_cast<std::size_t>(SIZEX) * static_cast<std::size_t>(SIZEY),
                                                 kDeadPixel),
             dirty_rows = std::vector<int>(),
             dirty_min_x = std::vector<int>(SIZEY, SIZEX),
             dirty_max_x = std::vector<int>(SIZEY, -1)](int) mutable {
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
                    SDL_UpdateTexture(texture,
                                      nullptr,
                                      pixels.data(),
                                      frame.view_width * static_cast<int>(sizeof(std::uint32_t)));
                } else if (!frame.changed->empty()) {
                    dirty_rows.clear();
                    for (const int index: *frame.changed) {
                        const Cell cell = {index % frame.stride - 1 - frame.origin_x,
                                           index / frame.stride - 1 - frame.origin_y};
                        if ((cell.first < 0) || (cell.first >= frame.view_width) ||
                            (cell.second < 0) || (cell.second >= frame.view_height)) {
                            continue;
                        }
                        const std::size_t pixel_index =
                                static_cast<std::size_t>(cell.second * frame.view_width + cell.first);
                        pixels[pixel_index] = (*frame.cells)[static_cast<std::size_t>(index)] != 0
                                              ? kAlivePixel
                                              : kDeadPixel;

                        if (dirty_max_x[static_cast<std::size_t>(cell.second)] < 0) {
                            dirty_rows.push_back(cell.second);
                            dirty_min_x[static_cast<std::size_t>(cell.second)] = cell.first;
                            dirty_max_x[static_cast<std::size_t>(cell.second)] = cell.first;
                        } else {
                            dirty_min_x[static_cast<std::size_t>(cell.second)] = std::min(
                                    dirty_min_x[static_cast<std::size_t>(cell.second)],
                                    cell.first);
                            dirty_max_x[static_cast<std::size_t>(cell.second)] = std::max(
                                    dirty_max_x[static_cast<std::size_t>(cell.second)],
                                    cell.first);
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
    );

    while (loop->run()) {
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}
