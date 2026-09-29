#include "hbr/runtime.hpp"
#include <stdexcept>
namespace hbr {
namespace {
bool terminal(State state) { return state == State::completed || state == State::cancelled || state == State::failed; }
}
bool Ticket::cancel() {
    std::lock_guard lock(data_->mutex);
    if (terminal(data_->state)) return false;
    data_->cancellation_requested = true;
    if (data_->state == State::queued) {
        data_->state = State::cancelled;
        data_->done.notify_all();
        return true;
    }
    return false;
}
State Ticket::state() const { std::lock_guard lock(data_->mutex); return data_->state; }
bool Ticket::cancellation_requested() const { std::lock_guard lock(data_->mutex); return data_->cancellation_requested; }
void Ticket::wait() const {
    std::unique_lock lock(data_->mutex);
    data_->done.wait(lock, [&] { return terminal(data_->state); });
    if (data_->error) std::rethrow_exception(data_->error);
}
Runtime::Runtime(std::size_t workers, std::size_t queue_capacity) : capacity_(queue_capacity) {
    if (workers < 1 || workers > 64 || !queue_capacity) throw std::invalid_argument("invalid runtime capacity");
    try {
        for (std::size_t i = 0; i < workers; ++i) workers_.emplace_back([this] { run(); });
    } catch (...) {
        { std::lock_guard lock(mutex_); closing_ = true; }
        available_.notify_all();
        for (auto& worker : workers_) worker.join();
        throw;
    }
}
Runtime::~Runtime() {
    { std::lock_guard lock(mutex_); closing_ = true; }
    available_.notify_all();
    for (auto& worker : workers_) worker.join();
}
Ticket Runtime::submit(std::function<void()> work) {
    if (!work) throw std::invalid_argument("empty work");
    auto state = std::make_shared<Ticket::Shared>();
    { std::lock_guard lock(mutex_);
      if (closing_) throw std::runtime_error("runtime is closing");
      if (queue_.size() >= capacity_) throw std::runtime_error("queue capacity reached");
      queue_.push_back({std::move(work), state}); }
    available_.notify_one();
    return Ticket(state);
}
void Runtime::run() {
    while (true) {
        Job job;
        { std::unique_lock lock(mutex_);
          available_.wait(lock, [&] { return closing_ || !queue_.empty(); });
          if (queue_.empty()) return;
          job = std::move(queue_.front()); queue_.pop_front(); }
        { std::lock_guard lock(job.status->mutex);
          if (job.status->state == State::cancelled) continue;
          job.status->state = State::running; }
        std::exception_ptr error;
        try { job.work(); } catch (...) { error = std::current_exception(); }
        // Destroy owned captures before publishing physical completion.
        job.work = {};
        { std::lock_guard lock(job.status->mutex);
          job.status->error = error;
          job.status->state = error ? State::failed : State::completed; }
        job.status->done.notify_all();
    }
}
}
