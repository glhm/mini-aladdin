#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

namespace mini_aladdin::bench {

class Timer {
public:
    using Clock    = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration  = std::chrono::nanoseconds;

    void start() noexcept { start_ = Clock::now(); }
    void stop()  noexcept { end_   = Clock::now(); }

    [[nodiscard]] double elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

    [[nodiscard]] double elapsed_ns() const noexcept {
        return static_cast<double>(
            std::chrono::duration_cast<Duration>(end_ - start_).count()
        );
    }

    [[nodiscard]] double throughput(std::size_t n_options) const noexcept {
        return static_cast<double>(n_options) / (elapsed_ms() * 1e-3);
    }

private:
    TimePoint start_{};
    TimePoint end_{};
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)) { timer_.start(); }
    
    ~ScopedTimer() {
        timer_.stop();
        std::cout << std::fixed << std::setprecision(3)
                  << "[" << name_ << "] " 
                  << timer_.elapsed_ms() << " ms\n";
    }

private:
    std::string name_;
    Timer timer_;
};

} // namespace mini_aladdin::bench
