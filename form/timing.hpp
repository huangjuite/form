#ifndef FORM_TIMING_HPP
#define FORM_TIMING_HPP

// TODO this imports basically everything...
#include <chrono>

namespace form {

struct Time {
  std::chrono::time_point<std::chrono::high_resolution_clock> m_time;

  constexpr static auto INVALID_TIME = decltype(m_time)::max();

  template <typename T = double>
  [[nodiscard]] constexpr T toSeconds() const noexcept {
    return std::chrono::duration<T>(m_time.time_since_epoch()).count();
  }

  [[nodiscard]] constexpr auto toNanoseconds() const noexcept {
    return std::chrono::nanoseconds(m_time.time_since_epoch()).count();
  }

  constexpr void
  setFromNanoseconds(const std::chrono::nanoseconds::rep nanoseconds) noexcept {
    m_time = decltype(m_time)(std::chrono::nanoseconds(nanoseconds));
  }

  template <typename T> constexpr void setFromSeconds(const T seconds) noexcept {
    m_time = decltype(m_time)(std::chrono::duration_cast<decltype(m_time)::duration>(
        std::chrono::duration<T>(seconds)));
  }

  [[nodiscard]] static constexpr Time min() noexcept {
    return {.m_time = decltype(m_time)::min()};
  }

  [[nodiscard]] static constexpr Time max() noexcept {
    return {.m_time = decltype(m_time)::max()};
  }

  [[nodiscard]] constexpr operator bool() const noexcept {
    return m_time != INVALID_TIME;
  }

  constexpr static Time Invalid() noexcept { return {INVALID_TIME}; }
};

[[nodiscard]] constexpr bool operator<(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time < rhs.m_time;
}

[[nodiscard]] constexpr bool operator<=(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time <= rhs.m_time;
}

[[nodiscard]] constexpr bool operator>(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time > rhs.m_time;
}

[[nodiscard]] constexpr bool operator>=(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time >= rhs.m_time;
}

[[nodiscard]] constexpr bool operator==(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time == rhs.m_time;
}

[[nodiscard]] constexpr bool operator!=(const Time lhs, const Time rhs) noexcept {
  return lhs.m_time != rhs.m_time;
}

struct Duration {
  std::chrono::high_resolution_clock::duration m_duration;
  constexpr static auto INVALID_DURATION = decltype(m_duration)::max();

  template <typename T = double>
  [[nodiscard]] constexpr T toSeconds() const noexcept {
    return std::chrono::duration<T>(m_duration).count();
  }

  [[nodiscard]] constexpr auto toNanoseconds() const noexcept {
    return std::chrono::nanoseconds(m_duration).count();
  }

  template <typename T> constexpr void setFromSeconds(const T seconds) noexcept {
    m_duration = std::chrono::duration_cast<decltype(m_duration)>(
        std::chrono::duration<T>(seconds));
  }

  template <typename T>
  constexpr void setFromNanoseconds(const T nanoseconds) noexcept {
    m_duration = std::chrono::duration_cast<decltype(m_duration)>(
        std::chrono::duration<T, std::chrono::nanoseconds::period>(nanoseconds));
  }

  template <typename T>
  [[nodiscard]] static constexpr Duration Nanoseconds(const T nanoseconds) noexcept {
    Duration duration;
    duration.setFromNanoseconds(nanoseconds);
    return duration;
  }

  template <typename T>
  [[nodiscard]] static constexpr Duration Seconds(const T seconds) noexcept {
    Duration duration;
    duration.setFromSeconds(seconds);
    return duration;
  }

  [[nodiscard]] constexpr operator bool() const noexcept {
    return m_duration != INVALID_DURATION;
  }

  [[nodiscard]] constexpr static Duration Invalid() noexcept {
    return {INVALID_DURATION};
  }
};

[[nodiscard]] constexpr Time operator+(const Duration lhs, const Time rhs) noexcept {
  return {.m_time = lhs.m_duration + rhs.m_time};
}

[[nodiscard]] constexpr Time operator+(const Time rhs, const Duration lhs) noexcept {
  return {.m_time = rhs.m_time + lhs.m_duration};
}

[[nodiscard]] constexpr Duration operator-(const Time lhs, const Time rhs) noexcept {
  return {.m_duration = lhs.m_time - rhs.m_time};
}

[[nodiscard]] constexpr bool operator<(const Duration lhs,
                                       const Duration rhs) noexcept {
  return lhs.m_duration < rhs.m_duration;
}

[[nodiscard]] constexpr bool operator<=(const Duration lhs,
                                        const Duration rhs) noexcept {
  return lhs.m_duration <= rhs.m_duration;
}

[[nodiscard]] constexpr bool operator>(const Duration lhs,
                                       const Duration rhs) noexcept {
  return lhs.m_duration > rhs.m_duration;
}

[[nodiscard]] constexpr bool operator>=(const Duration lhs,
                                        const Duration rhs) noexcept {
  return lhs.m_duration >= rhs.m_duration;
}

} // namespace form

#endif
