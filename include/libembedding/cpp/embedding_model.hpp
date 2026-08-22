/*
 * libembedding - cpp/embedding_model.hpp
 * C++ wrapper for dense text embedding.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_EMBEDDING_MODEL_HPP
#define LIBEMBEDDING_CPP_EMBEDDING_MODEL_HPP

#include <libembedding/text_embedding.h>
#include <libembedding/model_loader.h>
#include <libembedding/cpp/provider.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed {

struct EmbeddingOptions {
    std::string                  model_path;              /* HF name or local dir path */
    int                          threads = 0;
    int                          batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    int                          max_length = 0;
    bool                         show_download_progress = true;
    lembed_pooling_t             pooling = LEMBED_POOLING_MEAN;
    int                          dim = 0;
};

class EmbeddingModel {
public:
    EmbeddingModel(const std::string& model, const EmbeddingOptions& opts = {}) {
        lembed_text_options_t c_opts = lembed_text_options_default();

        c_opts.provider = parse_provider(opts.provider);
        c_opts.device_id = opts.device_id;
        c_opts.max_length = opts.max_length;
        c_opts.num_threads = opts.threads;
        c_opts.batch_size = opts.batch_size;
        c_opts.offline = opts.offline ? 1 : 0;
        c_opts.show_download_progress = opts.show_download_progress ? 1 : 0;
        if (!opts.cache_dir.empty()) {
            cache_dir_buf_ = opts.cache_dir;
            c_opts.cache_dir = cache_dir_buf_.c_str();
        }
        c_opts.dim = opts.dim;
        c_opts.pooling = opts.pooling;

        int idx = lembed_find_text_model_by_code(model.c_str());
        if (idx >= 0) {
            c_opts.model = (lembed_text_model_t)idx;
            check_or_throw(lembed_text_embedding_create(&c_opts, &ctx_));
        } else if (std::filesystem::exists(model)) {
            check_or_throw(lembed_text_embedding_create_from_path(
                model.c_str(), &c_opts, &ctx_));
        } else {
            throw std::invalid_argument("Unknown model or path: " + model);
        }

        dim_ = lembed_text_embedding_dim(ctx_);
    }

    ~EmbeddingModel() { close(); }

    EmbeddingModel(const EmbeddingModel&) = delete;
    EmbeddingModel& operator=(const EmbeddingModel&) = delete;

    EmbeddingModel(EmbeddingModel&& other) noexcept
        : ctx_(other.ctx_), dim_(other.dim_), cache_dir_buf_(std::move(other.cache_dir_buf_)) {
        other.ctx_ = nullptr;
    }

    EmbeddingModel& operator=(EmbeddingModel&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            dim_ = other.dim_;
            cache_dir_buf_ = std::move(other.cache_dir_buf_);
            other.ctx_ = nullptr;
        }
        return *this;
    }

    std::vector<std::vector<float>> embed(const std::vector<std::string>& texts,
                                          int batch_size = 0) {
        if (texts.empty()) return {};

        std::vector<const char*> c_texts;
        c_texts.reserve(texts.size());
        for (const auto& t : texts) c_texts.push_back(t.c_str());

        lembed_embeddings_t result = {0};
        check_or_throw(lembed_text_embedding_embed(
            ctx_, c_texts.data(), (int)texts.size(), batch_size, &result));

        std::vector<std::vector<float>> embeddings;
        try {
            embeddings.resize(result.num_embeddings);
            for (int i = 0; i < result.num_embeddings; i++) {
                embeddings[i].assign(
                    result.data + (size_t)i * result.dim,
                    result.data + (size_t)(i + 1) * result.dim);
            }
        } catch (...) {
            lembed_embeddings_free(&result);
            throw;
        }
        lembed_embeddings_free(&result);
        return embeddings;
    }

    int dimension() const {
        return ctx_ ? lembed_text_embedding_dim(ctx_) : 0;
    }

    lembed_model_desc_t info() const {
        lembed_model_desc_t d{};
        if (ctx_) {
            const lembed_model_desc_t* p = lembed_text_embedding_desc(ctx_);
            if (p) d = *p;
        }
        return d;
    }

    std::string name() const {
        if (!ctx_) return "";
        const char* n = lembed_text_embedding_model_name(ctx_);
        return n ? std::string(n) : "";
    }

    int max_length() const {
        return ctx_ ? lembed_text_embedding_max_length(ctx_) : 0;
    }

    void close() {
        if (ctx_) {
            lembed_text_embedding_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lembed_text_embedding_t* ctx_ = nullptr;
    int                       dim_ = 0;
    std::string               cache_dir_buf_;

    static void check_or_throw(lembed_status_t s) {
        if (s != LEMBED_OK) {
            throw std::runtime_error(
                std::string("libembedding error: ") +
                lembed_last_error());
        }
    }
};

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_EMBEDDING_MODEL_HPP */
