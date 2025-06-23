// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <thread>

class Watchdog
{
   public:
    constexpr static unsigned int WATCHDOG_GRANULARITY = 100;    // Check every 100 milliseconds
    constexpr static unsigned int WATCHDOG_DEF_TIMEOUT = 60000;  // Default timeout in milliseconds (1 minute)

    Watchdog(unsigned int milliseconds = 0, std::function<void()> callback = nullptr)
    {
        if (!milliseconds)
            milliseconds = WATCHDOG_DEF_TIMEOUT;
        Start(milliseconds, callback);
    }

    ~Watchdog() { Stop(); }

    void Start(unsigned int milliseconds, std::function<void()> callback = nullptr)
    {
        _interval = milliseconds;
        if (!callback)
        {
            _callback = []()
            {
                std::cerr << "Watchdog timeout! Application might be unresponsive." << std::endl;
                std::abort();
            };
        }
        else
            _callback = callback;
        Pet();
        _running = true;
        _thread = std::thread(&Watchdog::Loop, this);
    }

    void Stop()
    {
        _running = false;
        if (_thread.joinable())
        {
            _thread.join();
        }
    }

    void Pet() { _last_pet_time = std::chrono::steady_clock::now(); }

   private:
    unsigned int _interval;  // Watchdog timeout in milliseconds
    std::atomic<bool> _running;
    std::thread _thread;
    std::function<void()> _callback;
    std::chrono::steady_clock::time_point _last_pet_time;  // Last time the watchdog was pet

    void Loop()
    {
        while (_running)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - _last_pet_time).count();

            if (elapsed >= _interval)
            {
                _running = false;  // Stop the watchdog thread
                _callback();       // Trigger the timeout action
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(WATCHDOG_GRANULARITY));
        }
    }
};
