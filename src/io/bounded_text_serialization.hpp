#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>

namespace cosmo_nbody::io::detail {

// Two-pass serializer: count and enforce the bound first, then fill one exactly
// sized string. Both passes use std::ostream formatting to preserve bytes.
class CountingTextBuffer final : public std::streambuf {
public:
    explicit CountingTextBuffer(std::size_t maximum_bytes) noexcept
        : maximum_bytes_(maximum_bytes) {}

    std::size_t size() const noexcept { return size_; }

protected:
    std::streamsize xsputn(
        const char_type*,
        std::streamsize count) override {
        if (count < 0) {
            throw std::length_error(
                "Bounded text serializer received a negative write size");
        }
        const auto requested = static_cast<std::uintmax_t>(count);
        const auto remaining = static_cast<std::uintmax_t>(
            maximum_bytes_ - size_);
        if (requested > remaining) {
            throw std::length_error(
                "Bounded text serializer exceeds its admitted byte limit");
        }
        size_ += static_cast<std::size_t>(count);
        return count;
    }

    int_type overflow(int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof())) {
            return traits_type::not_eof(character);
        }
        if (size_ == maximum_bytes_) {
            throw std::length_error(
                "Bounded text serializer exceeds its admitted byte limit");
        }
        ++size_;
        return character;
    }

private:
    std::size_t maximum_bytes_{0};
    std::size_t size_{0};
};

class FixedTextBuffer final : public std::streambuf {
public:
    explicit FixedTextBuffer(std::string& output) noexcept
        : output_(output) {}

    std::size_t size() const noexcept { return size_; }

protected:
    std::streamsize xsputn(
        const char_type* source,
        std::streamsize count) override {
        if (count < 0) {
            throw std::length_error(
                "Fixed text serializer received a negative write size");
        }
        const auto requested = static_cast<std::uintmax_t>(count);
        const auto remaining = static_cast<std::uintmax_t>(
            output_.size() - size_);
        if (requested > remaining) {
            throw std::length_error(
                "Fixed text serializer exceeded its counted byte size");
        }
        const std::size_t bytes = static_cast<std::size_t>(count);
        if (bytes != 0U) {
            std::memcpy(output_.data() + size_, source, bytes);
        }
        size_ += bytes;
        return count;
    }

    int_type overflow(int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof())) {
            return traits_type::not_eof(character);
        }
        if (size_ == output_.size()) {
            throw std::length_error(
                "Fixed text serializer exceeded its counted byte size");
        }
        output_[size_++] = traits_type::to_char_type(character);
        return character;
    }

private:
    std::string& output_;
    std::size_t size_{0};
};

template <typename Renderer>
void render_text(std::streambuf& buffer, Renderer&& renderer) {
    std::ostream output(&buffer);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.precision(17);
    std::forward<Renderer>(renderer)(output);
}

template <typename Renderer>
std::string render_bounded_text(
    std::size_t maximum_bytes,
    Renderer&& renderer) {
    CountingTextBuffer counter(maximum_bytes);
    render_text(counter, renderer);

    std::string output(counter.size(), '\0');
    FixedTextBuffer sink(output);
    render_text(sink, std::forward<Renderer>(renderer));
    if (sink.size() != output.size()) {
        throw std::logic_error(
            "Bounded text serializer emitted inconsistent pass lengths");
    }
    return output;
}

} // namespace cosmo_nbody::io::detail
