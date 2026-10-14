#pragma once

#include <vector>

#include "tensor.hpp"

using tn::Tensor;

namespace kvcache {

class LayerKVCache {
   public:
    LayerKVCache(int layer_idx, int num_embed_dims)
        : layer_idx_(layer_idx), num_embed_dims_(num_embed_dims) {}
    LayerKVCache() = default;

    void add(Tensor& key, Tensor& value);
    void get_keys(Tensor& keys);
    void get_values(Tensor& values);
    int get_num_tokens();
    void remove_oldest_token();

   private:
    int layer_idx_;
    int num_embed_dims_;

    /* The key-value caches for each layer. Tokens x Embed Dimensions */
    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;
};

class KVCache {
   public:
    KVCache(int num_layers, int num_embed_dims)
        : num_layers_(num_layers), num_embed_dims_(num_embed_dims) {
        for (int i = 0; i < num_layers_; i++) {
            kv_cache_.emplace_back(i, num_embed_dims_);
        }
    }

    KVCache() = default;

    void add(int layer_idx, Tensor& key, Tensor& value);
    void get_keys(int layer_idx, Tensor& keys);
    void get_values(int layer_idx, Tensor& values);
    LayerKVCache& get_layer_kv_cache(int layer_idx);
    int get_num_tokens(int layer_idx);
    void remove_oldest_token();

   private:
    int num_layers_;
    int num_embed_dims_;

    /* The key-value caches for all layers. Layers x Tokens x Embed Dimensions */
    std::vector<LayerKVCache> kv_cache_;
};

}  // namespace kvcache