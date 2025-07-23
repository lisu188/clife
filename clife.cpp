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

#include <boost/range/irange.hpp>
#include <cstdlib>
#include <ctime>


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

static bool get_cell(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board, std::pair<int, int> coords) {
    return board->find(coords) != board->end();
}

static bool set_cell(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board, std::pair<int, int> coords,
                     bool val) {
    const std::unordered_set<std::pair<int, int>>::iterator it = board->find(coords);
    if (val && (it == board->end())) {
        board->insert(coords);
        return true;
    } else if (it != board->end()) {
        board->erase(coords);
        return true;
    }
    return false;
}

static bool flip_cell(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board, std::pair<int, int> pair) {
    return set_cell(board, pair, !get_cell(board, pair));
}

LifeBoard::LifeBoard(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board, int threads) : _threads(threads),
                                                                                                    _prev_board(
                                                                                                            board),
                                                                                                    _prev_diff(
                                                                                                            build_diff(
                                                                                                                    board)) {
}

int LifeBoard::adjacent_alive(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board,
                              std::pair<int, int> coords) const {
    int alive = 0;
    const std::shared_ptr<std::set<std::pair<int, int>>> ptr = getNext(coords);
    for (std::pair<int, int> next:*ptr) {
        if (get_cell(board, next)) {
            alive++;
        }
    }
    return alive;
}

bool LifeBoard::next_state(std::shared_ptr<std::unordered_set<std::pair<int, int>>> board,
                           std::pair<int, int> param) const {
    int nei = adjacent_alive(board, param);
    bool cell = get_cell(board, param);
    if (cell) {
        bool tmp = true;
        for (int it:survive) {
            tmp = tmp && (nei != it);
        }
        if (tmp) {
            return false;
        }
    } else {
        bool tmp = false;
        for (int it:born) {
            tmp = tmp || (nei == it);
        }
        if (tmp) {
            return true;
        }
    }
    return cell;
}

std::shared_ptr<std::list<std::pair<int, int>>> LifeBoard::iterate() {
    if (_iteration == 0) {
        _iteration++;
        return std::make_shared<std::list<std::pair<int, int>>>(_prev_board->begin(), _prev_board->end());
    }

    auto changed = std::make_shared<std::list<std::pair<int, int>>>();

    auto next_diff = std::make_shared<std::set<std::pair<int, int>>>();

    auto next_board = vstd::async([this]() {
        auto tmp = std::make_shared<std::unordered_set<std::pair<int, int>>>();
        tmp->insert(_next_board->begin(), _next_board->end());
        return tmp;
    });

    auto chain = std::make_shared<vstd::chain<std::pair<int, int>>>(
            [this, next_diff, next_board, changed](std::pair<int, int> crd) {
                changed->push_back(crd);
                flip_cell(next_board->get(), crd);
                next_diff->insert(crd);
                std::shared_ptr<std::set<std::pair<int, int>>> nxt = getNext(crd);
                next_diff->insert(nxt->begin(), nxt->end());
            });

    auto cb = [this, chain](int i) {
        return vstd::async([this, chain, i]() {
            // Determine the processing block size for each thread.
            size_t size = _prev_diff->size();
            size_t step = (size + _threads - 1) / _threads;

            // Calculate begin/end iterators for this thread's chunk.
            auto start = _prev_diff->begin();
            auto end = _prev_diff->begin();

            std::advance(start, step * i);

            // Clamp the end iterator so we never advance past _prev_diff->end().
            size_t end_pos = step * (i + 1);
            if (end_pos > size) {
                end_pos = size;
            }
            std::advance(end, end_pos);

            // Iterate through the assigned range.
            for (; start != end; ++start) {
                if (get_cell(_next_board, *start) != next_state(_prev_board, *start)) {
                    chain->invoke_async(*start);
                }
            }
        });
    };

    vstd::join(boost::irange(0, _threads) | boost::adaptors::transformed(cb))->thenAsync(
            [this, next_board, chain](std::set<void *>) {
                chain->terminate()->thenAsync([this, next_board]() {
                    _next_board = next_board->get();
                })->get();
            })->get();

    _prev_board.swap(_next_board);

    _prev_diff = next_diff;

    _iteration++;
    return changed;
}

LifeBoard::~LifeBoard() {

}

std::shared_ptr<std::set<std::pair<int, int>>> LifeBoard::getNext(std::pair<int, int> i) const {
    std::shared_ptr<std::set<std::pair<int, int>>> tmp = std::make_shared<std::set<std::pair<int, int>>>();
    int x = i.first;
    int y = i.second;
    tmp->insert(std::pair<int, int>(x - 1, y - 1));
    tmp->insert(std::pair<int, int>(x - 1, y));
    tmp->insert(std::pair<int, int>(x - 1, y + 1));
    tmp->insert(std::pair<int, int>(x, y - 1));
    tmp->insert(std::pair<int, int>(x, y + 1));
    tmp->insert(std::pair<int, int>(x + 1, y - 1));
    tmp->insert(std::pair<int, int>(x + 1, y));
    tmp->insert(std::pair<int, int>(x + 1, y + 1));
    return tmp;
}

std::shared_ptr<std::set<std::pair<int, int>>> LifeBoard::build_diff(
        std::shared_ptr<std::unordered_set<std::pair<int, int>>> board) {
    std::shared_ptr<std::set<std::pair<int, int>>> tmp = std::make_shared<std::set<std::pair<int, int>>>();
    for (std::pair<int, int> coords:*board) {
        tmp->insert(coords);
        std::shared_ptr<std::set<std::pair<int, int>>> ptr = getNext(coords);
        tmp->insert(ptr->begin(), ptr->end());
    }
    return tmp;
}

int main(int argc, char **args) {
    srand(time(0));
    int SIZEX = 500;
    int SIZEY = 500;
    float scale = 1;
    float factor = 0.8;
    int seeds = (int) (SIZEX * SIZEY * factor);
    std::shared_ptr<std::unordered_set<std::pair<int, int>>> tmp = std::make_shared<std::unordered_set<std::pair<int, int>>>(
            seeds);
    for (int i = 0; i < seeds; i++) {
        flip_cell(tmp, std::pair<int, int>(rand() % SIZEX, rand() % SIZEY));
    }
    std::shared_ptr<LifeBoard> board = std::make_shared<LifeBoard>(tmp, 16);

    SDL_Window *window = 0;
    SDL_Renderer *renderer = 0;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(SIZEX * scale, SIZEY * scale, SDL_WINDOW_OPENGL, &window, &renderer);

    vstd::async([board, window, renderer, scale, SIZEX, SIZEY]() {
        while (true) {
            auto data = board->iterate();
            vstd::later([data, window, renderer, scale, SIZEX, SIZEY]() {
                SDL_RenderClear(renderer);
                SDL_Surface *surface = SDL_CreateRGBSurface(0, SIZEX * scale, SIZEY * scale, 32, 0, 0, 0, 0);
                for (std::pair<int, int> changed:*data) {
                    SDL_Rect rect;
                    rect.x = changed.first * scale;
                    rect.y = changed.second * scale;
                    rect.w = scale;
                    rect.h = scale;
                    SDL_FillRect(surface, &rect, SDL_MapRGB(surface->format, 255, 255, 255));
                }
                SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
                SDL_RenderCopy(renderer, texture, NULL, NULL);
                SDL_RenderPresent(renderer);
                SDL_DestroyTexture(texture);
                SDL_FreeSurface(surface);
            });
        }
    });

    while (vstd::event_loop<>::instance()->run());

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}


