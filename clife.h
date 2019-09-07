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
#pragma once

#include <SDL.h>
#include <unordered_set>
#include <vstd.h>


class LifeBoard {
public:
    LifeBoard(std::shared_ptr<std::unordered_set<std::pair<int, int >>> board, int threads);

    ~LifeBoard();

    std::shared_ptr<std::list<std::pair<int, int >>> iterate();

    std::shared_ptr<std::unordered_set<std::pair<int, int >>> _prev_board = std::make_shared<std::unordered_set<std::pair<int, int >>>();
    std::shared_ptr<std::unordered_set<std::pair<int, int >>> _next_board = std::make_shared<std::unordered_set<std::pair<int, int >>>();

    std::shared_ptr<std::set<std::pair<int, int >>> _prev_diff = std::make_shared<std::set<std::pair<int, int >>>();

    bool next_state(std::shared_ptr<std::unordered_set<std::pair<int, int >>> board,
                    std::pair<int, int> param) const;

    int _iteration = 0;

    const int _threads;

    int adjacent_alive(std::shared_ptr<std::unordered_set<std::pair<int, int >>> board,
                       std::pair<int, int> coords) const;

    std::set<int> survive = {2, 3};
    std::set<int> born = {3};

    std::shared_ptr<std::set<std::pair<int, int >>> getNext(std::pair<int, int> i) const;

    std::shared_ptr<std::set<std::pair<int, int >>> build_diff(
            std::shared_ptr<std::unordered_set<std::pair<int, int >>> shared_ptr);
};

