// Phase 17: generic ncnn FP32 VAE stage runner.
// Usage:
//   vae_stage_runner <model.ncnn.param> <model.ncnn.bin> <input.bin> <output.bin> <in_dim0> <in_dim1> ... <out_dim0> <out_dim1> ...
// Reads a raw float32 input blob with the given dims (innermost contiguous),
// runs the ncnn net (fp32, CPU), writes the raw float32 output blob with the
// given output dims.  Returns 0 on success, 1 on runtime mismatch.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "net.h"

static bool read_file(const std::string& path, std::vector<float>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz / sizeof(float));
    size_t rd = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return rd == out.size();
}

static bool write_file(const std::string& path, const float* data, size_t n)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    size_t wr = fwrite(data, sizeof(float), n, f);
    fclose(f);
    return wr == n;
}

int main(int argc, char** argv)
{
    if (argc < 9)
    {
        fprintf(stderr,
                "usage: vae_stage_runner <param> <bin> <input.bin> <output.bin> "
                "<in_dim0> <in_dim1> ... <out_dim0> <out_dim1> ...\n");
        return 2;
    }
    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string input_path = argv[3];
    const std::string output_path = argv[4];

    std::vector<int> in_dims;
    std::vector<int> out_dims;
    int arg = 5;
    for (; arg < argc; ++arg)
    {
        if (strcmp(argv[arg], "--") == 0)
        {
            ++arg;
            break;
        }
        if (strcmp(argv[arg], "--extract") == 0)
        {
            arg += 1;  // skip blob-name list; --outdir handled below
            continue;
        }
        if (strcmp(argv[arg], "--outdir") == 0)
        {
            ++arg;  // skip dir
            continue;
        }
        in_dims.push_back(atoi(argv[arg]));
    }
    for (; arg < argc; ++arg)
        out_dims.push_back(atoi(argv[arg]));
    if (in_dims.empty() || out_dims.empty())
    {
        fprintf(stderr, "must give both input and output dims separated by --\n");
        return 2;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;

    int rc = net.load_param(param_path.c_str());
    if (rc != 0)
    {
        fprintf(stderr, "load_param failed rc=%d\n", rc);
        return 3;
    }
    rc = net.load_model(bin_path.c_str());
    if (rc != 0)
    {
        fprintf(stderr, "load_model failed rc=%d\n", rc);
        return 4;
    }

    std::vector<float> input;
    if (!read_file(input_path, input))
    {
        fprintf(stderr, "cannot read input %s\n", input_path.c_str());
        return 5;
    }

    ncnn::Mat in;
    if (in_dims.size() == 1)
        in = ncnn::Mat(in_dims[0], input.data());
    else if (in_dims.size() == 2)
        in = ncnn::Mat(in_dims[0], in_dims[1], input.data());
    else if (in_dims.size() == 3)
        in = ncnn::Mat(in_dims[0], in_dims[1], in_dims[2], input.data());
    else if (in_dims.size() == 4)
        in = ncnn::Mat(in_dims[0], in_dims[1], in_dims[2], in_dims[3], input.data());
    else
    {
        fprintf(stderr, "unsupported input rank %d (pass 5D as w,h,d,c with batch removed)\\n", (int)in_dims.size());
        return 6;
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    rc = ex.extract("out0", out);
    if (rc != 0)
    {
        fprintf(stderr, "extract failed rc=%d\n", rc);
        return 7;
    }

    if (out.dims != (int)out_dims.size())
    {
        fprintf(stderr, "output rank mismatch: ncnn=%d expected=%d\n", out.dims, (int)out_dims.size());
        return 8;
    }
    long long expected = 1;
    for (int d : out_dims)
        expected *= d;
    const size_t contig_total = (size_t)out.w * out.h * out.d * out.c;
    if ((long long)contig_total != expected)
    {
        fprintf(stderr, "output size mismatch: ncnn=%zu expected=%lld\n", contig_total, expected);
        return 9;
    }

    // ncnn 4D Mats may carry cstep alignment padding; materialize contiguous.
    const size_t plane = (size_t)out.w * out.h * out.d;
    std::vector<float> contig(plane * (size_t)out.c);
    if (out.cstep == plane)
    {
        memcpy(contig.data(), (const float*)out.data, plane * out.c * sizeof(float));
    }
    else
    {
        const float* src = (const float*)out.data;
        for (int q = 0; q < out.c; ++q)
        {
            memcpy(contig.data() + q * plane, src + q * out.cstep, plane * sizeof(float));
            src += out.cstep;
        }
    }
    if (!write_file(output_path, contig.data(), contig.size()))
    {
        fprintf(stderr, "cannot write output %s\n", output_path.c_str());
        return 10;
    }

    // optional intermediate blob dump (--extract name1,name2 --outdir dir)
    for (int i = 5; i < argc; ++i)
    {
        if (strcmp(argv[i], "--extract") == 0 && i + 3 < argc)
        {
            const std::string outdir = argv[i + 3];
            std::string names = argv[i + 1];
            size_t pos = 0;
            while (pos <= names.size())
            {
                size_t comma = names.find(',', pos);
                std::string name = names.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!name.empty())
                {
                    ncnn::Mat blob;
                    int rc2 = ex.extract(name.c_str(), blob);
                    if (rc2 != 0)
                    {
                        fprintf(stderr, "extract blob %s failed rc=%d\n", name.c_str(), rc2);
                        return 11;
                    }
                    const size_t total = (size_t)blob.w * blob.h * blob.d * blob.c;
                    std::vector<float> buf(total);
                    if (blob.dims == 1)
                    {
                        memcpy(buf.data(), (const float*)blob.data, total * sizeof(float));
                    }
                    else
                    {
                        const size_t plane = (size_t)blob.w * blob.h * blob.d;
                        const float* src = (const float*)blob.data;
                        for (int q = 0; q < blob.c; ++q)
                        {
                            memcpy(buf.data() + q * plane, src + q * blob.cstep, plane * sizeof(float));
                            src += blob.cstep;
                        }
                    }
                    std::string path = outdir + "\\" + name + ".bin";
                    FILE* f = fopen(path.c_str(), "wb");
                    if (!f)
                    {
                        fprintf(stderr, "cannot write %s\n", path.c_str());
                        return 12;
                    }
                    fwrite(buf.data(), sizeof(float), buf.size(), f);
                    fclose(f);
                    fprintf(stdout, "dumped %s total=%d\n", name.c_str(), (int)total);
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            i += 2;
        }
    }

    fprintf(stdout, "ok total=%d\n", (int)contig.size());
    fprintf(stdout, "\n");
    return 0;
}
