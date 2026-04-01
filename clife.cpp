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
#include <future>
#include <iostream>
#include <unordered_set>
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
    using CellList = LifeBoard::CellList;
    constexpr std::size_t kMinParallelDiffSize = 4096;

    constexpr std::array<Cell, 8> kNeighborOffsets = {{
            {-1, -1}, {-1, 0}, {-1, 1},
            {0,  -1},          {0,  1},
            {1,  -1}, {1,  0}, {1,  1},
    }};

    struct WorkerResult {
        CellList changed;
        CellSet next_diff;
    };

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

    void add_diff_cells(CellSet &diff, const Cell &cell) {
        diff.insert(cell);
        for (const Cell &offset: kNeighborOffsets) {
            diff.insert({cell.first + offset.first, cell.second + offset.second});
        }
    }
}

LifeBoard::LifeBoard(std::shared_ptr<CellSet> board, int threads) : _threads(std::max(1, threads)),
                                                                    _prev_board(std::move(board)),
                                                                    _prev_diff(build_diff(*_prev_board)) {
    _next_board->reserve(_prev_board->size());
}

int LifeBoard::adjacent_alive(const CellSet &board, Cell coords) const {
    int alive = 0;
    for (const Cell &offset: kNeighborOffsets) {
        if (get_cell(board, {coords.first + offset.first, coords.second + offset.second})) {
            alive++;
        }
    }
    return alive;
}

bool LifeBoard::next_state(const CellSet &board, Cell param) const {
    const int neighbors = adjacent_alive(board, param);
    return neighbors == 3 || (get_cell(board, param) && neighbors == 2);
}

std::shared_ptr<const CellSet> LifeBoard::iterate() {
    if (_iteration == 0) {
        _iteration++;
        return _prev_board;
    }

    auto next_board = std::make_shared<CellSet>(*_next_board);
    next_board->reserve(std::max(_next_board->size(), _prev_board->size()) + _prev_diff.size());

    if (_prev_diff.empty()) {
        _next_board = next_board;
        _prev_board.swap(_next_board);
        _iteration++;
        return _prev_board;
    }

    const size_t worker_count = _prev_diff.size() < kMinParallelDiffSize
                                ? 1
                                : std::min(_prev_diff.size(), static_cast<size_t>(_threads));
    const size_t step = (_prev_diff.size() + worker_count - 1) / worker_count;

    auto process_chunk = [this](size_t begin, size_t end) {
        WorkerResult result;
        result.changed.reserve(end - begin);
        result.next_diff.reserve((end - begin) * (kNeighborOffsets.size() + 1));

        const CellSet &current_board = *_prev_board;
        const CellSet &older_board = *_next_board;
        for (size_t index = begin; index < end; ++index) {
            const Cell &cell = _prev_diff[index];
            if (get_cell(older_board, cell) != next_state(current_board, cell)) {
                result.changed.push_back(cell);
                add_diff_cells(result.next_diff, cell);
            }
        }

        return result;
    };

    CellSet merged_next_diff;
    merged_next_diff.reserve(_prev_diff.size() * 2 + 1);

    if (worker_count == 1) {
        WorkerResult result = process_chunk(0, _prev_diff.size());
        for (const Cell &cell: result.changed) {
            flip_cell(*next_board, cell);
        }
        merged_next_diff.insert(result.next_diff.begin(), result.next_diff.end());
    } else {
        std::vector<std::future<WorkerResult>> futures;
        futures.reserve(worker_count);

        for (size_t worker = 0; worker < worker_count; ++worker) {
            const size_t begin = worker * step;
            const size_t end = std::min(begin + step, _prev_diff.size());
            if (begin >= end) {
                break;
            }
            futures.push_back(std::async(std::launch::async, process_chunk, begin, end));
        }

        for (std::future<WorkerResult> &future: futures) {
            WorkerResult result = future.get();
            for (const Cell &cell: result.changed) {
                flip_cell(*next_board, cell);
            }
            merged_next_diff.insert(result.next_diff.begin(), result.next_diff.end());
        }
    }

    _prev_diff.assign(merged_next_diff.begin(), merged_next_diff.end());
    _next_board = next_board;
    _prev_board.swap(_next_board);

    _iteration++;
    return _prev_board;
}

LifeBoard::~LifeBoard() {

}

LifeBoard::CellList LifeBoard::build_diff(const CellSet &board) const {
    CellSet diff;
    diff.reserve(board.size() * (kNeighborOffsets.size() + 1));
    for (const Cell &coords: board) {
        add_diff_cells(diff, coords);
    }
    return CellList(diff.begin(), diff.end());
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
    std::shared_ptr<LifeBoard> board = std::make_shared<LifeBoard>(tmp, 16);

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

    auto loop = vstd::event_loop<>::instance();
    loop->registerFrameCallback(
            [board, renderer, scale, points = std::vector<SDL_Point>(), rects = std::vector<SDL_Rect>()](int) mutable {
        const auto data = board->iterate();
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        if (scale == 1.0f) {
            points.clear();
            points.reserve(data->size());
            for (const Cell &cell: *data) {
                points.push_back(SDL_Point{cell.first, cell.second});
            }
            if (!points.empty()) {
                SDL_RenderDrawPoints(renderer, points.data(), static_cast<int>(points.size()));
            }
        } else {
            rects.clear();
            rects.reserve(data->size());
            for (const Cell &cell: *data) {
                rects.push_back(SDL_Rect{
                        static_cast<int>(cell.first * scale),
                        static_cast<int>(cell.second * scale),
                        static_cast<int>(scale),
                        static_cast<int>(scale),
                });
            }
            if (!rects.empty()) {
                SDL_RenderFillRects(renderer, rects.data(), static_cast<int>(rects.size()));
            }
        }

        SDL_RenderPresent(renderer);
    });

    while (loop->run()) {
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}
