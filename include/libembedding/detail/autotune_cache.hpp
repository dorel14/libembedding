/*
 * libembedding - detail/autotune_cache.hpp
 * Auto-tuning cache system
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP

#include "libembedding/autotuner.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lembed { namespace detail {

/* Get number of logical CPU cores */
inline int cpu_logical_cores() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/* Get physical CPU cores */
inline int cpu_physical_cores() {
    int logical = cpu_logical_cores();
    int physical = logical;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len > 0) {
        auto* buf = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
            int n_phys = 0;
            uint8_t* ptr = (uint8_t*)buf;
            uint8_t* end = ptr + len;
            while (ptr < end) {
                auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
                if (info->Relationship == RelationProcessorCore) {
                    n_phys++;
                }
                ptr += info->Size;
            }
            if (n_phys > 0) physical = n_phys;
        }
        free(buf);
    }
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        int max_phys = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "cpu cores", 9) == 0) {
                int n = 0;
                if (sscanf(line, "cpu cores : %d", &n) == 1 && n > max_phys)
                    max_phys = n;
            }
        }
        if (max_phys > 0) physical = max_phys;
        fclose(f);
    }
#endif
    return physical;
}

/* Get CPU brand string */
inline std::string cpu_brand_string() {
    char cpu_brand[64] = "unknown";
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD size = sizeof(cpu_brand);
        RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
                        (LPBYTE)cpu_brand, &size);
        RegCloseKey(hKey);
    }
#else
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    strncpy(cpu_brand, colon + 2, sizeof(cpu_brand) - 1);
                    cpu_brand[sizeof(cpu_brand) - 1] = '\0';
                    char* nl = strchr(cpu_brand, '\n');
                    if (nl) *nl = '\0';
                }
                break;
            }
        }
        fclose(fp);
    }
#endif
    /* Sanitize */
    for (int i = 0; cpu_brand[i]; i++) {
        char c = cpu_brand[i];
        if (c == ' ' || c == '(' || c == ')' || c == '@' || c == '.' || c == '/')
            cpu_brand[i] = '_';
    }
    return std::string(cpu_brand);
}

/* =========================================================================
 * Cache system for autotune results
 * ========================================================================= */

inline std::string autotune_cache_dir() {
#ifdef _WIN32
    const char* local_appdata = std::getenv("LOCALAPPDATA");
    if (local_appdata) {
        return std::string(local_appdata) + "\\libembedding\\autotune";
    }
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        return std::string(userprofile) + "\\AppData\\Local\\libembedding\\autotune";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.cache/libembedding/autotune";
    }
#endif
    return "./libembedding_autotune_cache";
}

/* Build cache key from model_name + hardware + versions */
inline std::string get_cache_key(const char* model_name) {
    std::string brand = cpu_brand_string();
    int logical = cpu_logical_cores();
    int physical = cpu_physical_cores();

    /* Sanitize model name */
    std::string safe_model = model_name;
    for (auto& c : safe_model) {
        if (c == '/' || c == '\\' || c == ':' || c == ' ')
            c = '_';
    }

    /* ORT version (major.minor only) */
    const OrtApiBase* ort_base = OrtGetApiBase();
    const char* ort_ver = ort_base->GetVersionString();
    int ort_major = 0, ort_minor = 0;
    if (ort_ver) {
        sscanf(ort_ver, "%d.%d", &ort_major, &ort_minor);
    }

    std::ostringstream key;
    key << logical << "x" << physical << "_" << brand << "_" << safe_model
        << "_ort" << ort_major << "." << ort_minor
        << "_v" << LIBEMBEDDING_VERSION_MAJOR << "." << LIBEMBEDDING_VERSION_MINOR;
    return key.str();
}

/* Get cache file path for a model */
inline std::string get_cache_path(const char* model_name, const std::string& subdir = "") {
    std::string dir = autotune_cache_dir();
    if (!subdir.empty()) dir += "/" + subdir;
    std::filesystem::create_directories(dir);
    return dir + "/" + get_cache_key(model_name) + ".json";
}

/* Trim whitespace and quotes from JSON values */
inline void trim_json_value(std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == ','))
        s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '"' || s.back() == ','))
        s.pop_back();
}

/* Generic JSON cache reader */
template <typename T>
bool read_cache_json(const std::string& path, T& result,
                     const std::vector<std::pair<std::string, double T::*>>& fields) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        trim_json_value(key);
        trim_json_value(val);

        for (const auto& field : fields) {
            if (key == field.first) {
                if (field.second == nullptr) continue; /* skip non-numeric */
                result.*field.second = std::stod(val);
                break;
            }
        }
    }
    return true;
}

/* Generic JSON cache writer */
template <typename T>
void write_cache_json(const std::string& path, const T& result,
                      const std::vector<std::pair<std::string, double T::*>>& fields) {
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "{\n";
    bool first = true;
    for (const auto& field : fields) {
        if (!first) f << ",\n";
        f << "  \"" << field.first << "\": " << result.*field.second;
        first = false;
    }
    f << "\n}\n";
}

/* Clear cache for a model (or all if model_name is nullptr) */
inline void clear_autotune_cache(const char* model_name, const std::string& subdir = "") {
    if (model_name) {
        std::string path = get_cache_path(model_name, subdir);
        try { std::filesystem::remove(path); } catch (...) {}
    } else {
        std::string dir = autotune_cache_dir();
        if (!subdir.empty()) dir += "/" + subdir;
        if (std::filesystem::exists(dir)) {
            try { std::filesystem::remove_all(dir); } catch (...) {}
        }
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP */
