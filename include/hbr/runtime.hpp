#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace hbr {
enum class State { queued, running, completed, cancelled, failed };
class Ticket {
public:
    struct Shared {
        mutable std::mutex mutex;
        std::condition_variable done;
        State state = State::queued;
        bool cancellation_requested = false;
        std::exception_ptr error;
    };
    explicit Ticket(std::shared_ptr<Shared> data) : data_(std::move(data)) {}
    bool cancel();
    State state() const;
    bool cancellation_requested() const;
    void wait() const;
private:
    std::shared_ptr<Shared> data_;
};
class Runtime {
public:
    explicit Runtime(std::size_t workers = 2, std::size_t queue_capacity = 16);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Ticket submit(std::function<void()> work);
private:
    struct Job { std::function<void()> work; std::shared_ptr<Ticket::Shared> status; };
    void run();
    std::mutex mutex_;
    std::condition_variable available_;
    std::deque<Job> queue_;
    std::vector<std::thread> workers_;
    std::size_t capacity_;
    bool closing_ = false;
};
}
