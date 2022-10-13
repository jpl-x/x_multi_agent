#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <functional>

template<class T>
class MessageQueue {
public:
    void send(T &&message) {
        std::lock_guard<std::mutex> lck(_mtx);
        _queue.push_back(std::move(message));
        _cond.notify_one();
    }

    T receive() {
        std::unique_lock<std::mutex> lck(_mtx);
        _cond.wait(lck, [this] { return !_queue.empty(); });

        T msg = std::move(_queue.back());
        _queue.pop_back();

        return msg;
    }

    void readAll(const std::function<void(T)> &f) {
        std::unique_lock<std::mutex> lck(_mtx);
        if (lck.try_lock()) {
            while (!_queue.empty()) {
                T msg = std::move(_queue.back());
                _queue.pop_back();
                f(msg);
            }
        }
    }

private:
    std::deque<T> _queue;
    std::condition_variable _cond;
    std::mutex _mtx;
};