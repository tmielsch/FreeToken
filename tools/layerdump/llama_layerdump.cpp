// llama_layerdump.cpp
// Per-layer residual dump tool for the Loop-Divergence layer harness (FreeToken vs llama.cpp).
//
// Runs a single prefill of the given token ids on the CPU reference and dumps, for every
// layer boundary, the FULL hyper-connection residual (hc * n_embd floats per token) plus
// the last-token logits row. Uses the fork's per-layer extraction API
// (llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp) whose buffer sizing
// was patched to n_embd_out so the wide residual survives the copy.
//
// Usage:
//   layerdump -m model.gguf -i tokens.txt -o outdir
// where tokens.txt is comma/space/newline separated token ids (byte-exact on both sides).
//
// Outputs (raw f32 little-endian):
//   outdir/meta.json
//   outdir/layer_%03d.f32   n_tokens * n_embd_out floats, row-major [token][stream*hidden + d]
//   outdir/logits_last.f32  n_vocab floats (logits of the last prefill token)
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

static void usage(FILE * f) {
    fprintf(f, "usage: llama-layerdump -m model.gguf -i tokens.txt -o outdir [-t threads] [-c n_ctx]\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string ids_path;
    std::string out_dir;
    bool f32_dense = false;
    int n_threads = 30;
    int n_ctx = 4096;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) { ids_path = argv[++i]; }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) { out_dir = argv[++i]; }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) { n_ctx = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-f32") == 0) { f32_dense = true; }
        else { usage(stderr); return 1; }
    }
    if (model_path.empty() || ids_path.empty() || out_dir.empty()) {
        usage(stderr);
        return 2;
    }

    std::vector<llama_token> tokens;
    {
        std::ifstream f(ids_path, std::ios::in);
        if (!f) { fprintf(stderr, "cannot open %s\n", ids_path.c_str()); return 3; }
        int id;
        while (f >> id) {
            tokens.push_back((llama_token) id);
        }
    }
    fprintf(stderr, "tokens: %zu\n", tokens.size());
    if (tokens.empty()) { fprintf(stderr, "no tokens\n"); return 4; }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;           // CPU reference (-ngl 0), matching the harness

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 5; }

    const int n_layer    = llama_model_n_layer(model);
    const int n_embd     = llama_model_n_embd(model);
    const int n_embd_out = llama_model_n_embd_out(model);
    const int n_vocab    = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int hc         = n_embd_out / n_embd;
    fprintf(stderr, "n_layer=%d n_embd=%d n_embd_out=%d hc=%d n_vocab=%d\n",
            n_layer, n_embd, n_embd_out, hc, n_vocab);

    if (n_embd_out % n_embd != 0) {
        fprintf(stderr, "unexpected geometry n_embd_out=%d n_embd=%d\n", n_embd_out, n_embd);
        return 6;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = n_ctx;
    cparams.n_threads        = n_threads;
    cparams.n_threads_batch  = n_threads;
    cparams.n_batch          = (int) tokens.size(); // one ubatch so the residual rows are contiguous
    cparams.f32_dense_mm     = f32_dense;           // A/B: fp32 weights => fp32 activations in dense GGUF matmuls
    if (cparams.n_batch < 1) cparams.n_batch = 1;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 7; }

    // Enable per-layer extraction for every boundary incl. after the last layer, plus the
    // extra op states stored at padded t_layer_inp indices (see qwen4exp.cpp):
    //   residual: lid il (0 .. n_layer), attn_out: n_layer+1+il, mlp_in: 2*(n_layer+1)+il,
    //   mlp_out: 3*(n_layer+1)+il, attn_in: 4*(n_layer+1)+il
    for (int il = 0; il <= n_layer; ++il) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
    }
    for (int il = 0; il < n_layer; ++il) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) (n_layer + 1 + il), true);
        llama_set_embeddings_layer_inp(ctx, (uint32_t) (2 * (n_layer + 1) + il), true);
        llama_set_embeddings_layer_inp(ctx, (uint32_t) (3 * (n_layer + 1) + il), true);
        llama_set_embeddings_layer_inp(ctx, (uint32_t) (4 * (n_layer + 1) + il), true);
    }

    // single prefill decode of the whole prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (int rc = llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return 8;
    }

    std::string outdir_s = out_dir;
    if (outdir_s.back() == '/' || outdir_s.back() == '\\') outdir_s.pop_back();

    // meta.json
    {
        std::ofstream f(outdir_s + "/meta.json", std::ios::out);
        f << "{"
          << "\"n_layer\":"   << n_layer    << ","
          << "\"n_tokens\":"  << (int) tokens.size() << ","
          << "\"n_embd\":"    << n_embd     << ","
          << "\"n_embd_out\":" << n_embd_out << ","
          << "\"hc\":"        << hc         << ","
          << "\"n_vocab\":"   << n_vocab    << ","
          << "\"f32_dense\":" << (f32_dense ? "true" : "false") << ","
          << "\"has_ops\":true"
          << "}\n";
    }

    // per-layer residual rows: [n_tokens][n_embd_out] f32
    for (int il = 0; il <= n_layer; ++il) {
        const float * data = llama_get_embeddings_layer_inp(ctx, (uint32_t) il);
        if (!data) {
            fprintf(stderr, "layer %d: no data\n", il);
            return 9;
        }
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.f32", outdir_s.c_str(), il);
        std::ofstream f(path, std::ios::binary | std::ios::out);
        f.write(reinterpret_cast<const char *>(data), (std::streamsize)(tokens.size() * n_embd_out * sizeof(float)));
    }

    // per-layer op states: [n_tokens][n_embd] f32 at the padded slots
    const char * op_names[4] = { "attn_out", "mlp_in", "mlp_out", "attn_in" };
    for (int oi = 0; oi < 4; ++oi) {
        for (int il = 0; il < n_layer; ++il) {
            const uint32_t lid = (uint32_t) ((oi + 1) * (n_layer + 1) + il);
            const float * data = llama_get_embeddings_layer_inp(ctx, lid);
            if (!data) {
                fprintf(stderr, "%s layer %d: no data\n", op_names[oi], il);
                return 11;
            }
            char path[512];
            snprintf(path, sizeof(path), "%s/%s_%03d.f32", outdir_s.c_str(), op_names[oi], il);
            std::ofstream f(path, std::ios::binary | std::ios::out);
            f.write(reinterpret_cast<const char *>(data), (std::streamsize)(tokens.size() * n_embd * sizeof(float)));
        }
    }

    // last-token logits row (n_vocab f32). llama_get_logits returns the last-output row.
    {
        const float * lg = llama_get_logits(ctx);
        if (!lg) { fprintf(stderr, "no logits\n"); return 10; }
        std::ofstream f(outdir_s + "/logits_last.f32", std::ios::binary | std::ios::out);
        f.write(reinterpret_cast<const char *>(lg), (std::streamsize)(n_vocab * sizeof(float)));
    }

    llama_free(ctx);
    llama_model_free(model);
    fprintf(stderr, "done -> %s\n", outdir_s.c_str());
    return 0;
}
