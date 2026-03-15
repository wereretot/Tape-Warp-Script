#pragma once
// ── Self-contained ImGui File Browser ────────────────────────────────────────
#include <imgui.h>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <cctype>

namespace fs = std::filesystem;

// ── Recent files list ─────────────────────────────────────────────────────────
class RecentFiles {
public:
    static constexpr int MAX = 12;
    void load(const std::string& path) {
        _path = path; _items.clear();
        std::ifstream f(path); std::string line;
        while (std::getline(f, line) && (int)_items.size() < MAX)
            if (!line.empty()) _items.push_back(line);
    }
    void save() {
        if (_path.empty()) return;
        std::ofstream f(_path);
        for (auto& s : _items) f << s << "\n";
    }
    void push(const std::string& p) {
        _items.erase(std::remove(_items.begin(),_items.end(),p),_items.end());
        _items.insert(_items.begin(), p);
        if ((int)_items.size() > MAX) _items.resize(MAX);
        save();
    }
    const std::vector<std::string>& items() const { return _items; }
private:
    std::vector<std::string> _items;
    std::string _path;
};

// ── File dialog ───────────────────────────────────────────────────────────────
class FileDialog {
public:
    enum class Mode { Load, Save };
    static inline RecentFiles recents;

    void open_load(const char* title,
                   std::vector<std::string> extensions = {},
                   const std::string& start_dir = "")
    {
        _title      = title;
        _mode       = Mode::Load;
        _extensions = std::move(extensions);
        _open       = true;
        _result.clear();
        _filename_buf[0] = '\0';
        _error.clear();
        _show_all = false;
        _navigate(start_dir.empty() ? _home() : start_dir);
    }

    void open_save(const char* title,
                   const std::string& default_name = "",
                   const std::string& start_dir = "")
    {
        _title      = title;
        _mode       = Mode::Save;
        _extensions.clear();
        _open       = true;
        _result.clear();
        _error.clear();
        _show_all = true;
        std::strncpy(_filename_buf, default_name.c_str(), sizeof(_filename_buf)-1);
        _navigate(start_dir.empty() ? _home() : start_dir);
    }

    // Returns true when dismissed. result() is non-empty on confirm.
    bool draw() {
        if (!_open) return false;

        ImGui::SetNextWindowSize({800, 560}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(
            ImGui::GetMainViewport()->GetCenter(),
            ImGuiCond_FirstUseEver, {0.5f, 0.5f});

        bool dismissed = false;
        if (!ImGui::Begin(_title.c_str(), &_open, ImGuiWindowFlags_NoCollapse)) {
            ImGui::End();
            if (!_open) dismissed = true;
            return dismissed;
        }

        // ── Path bar ─────────────────────────────────────────────────────────
        {
            static char path_edit[1024];
            if (!_editing_path) {
                std::strncpy(path_edit, _current_dir.string().c_str(), sizeof(path_edit)-1);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f,.85f,0.f,1.f));
            if (ImGui::InputText("##pathbar", path_edit, sizeof(path_edit),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
                _navigate(path_edit);
                _editing_path = false;
            }
            if (ImGui::IsItemActive()) _editing_path = true;
            else                       _editing_path = false;
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("^ Up")) _go_up();
        }

        // ── Breadcrumbs ───────────────────────────────────────────────────────
        {
            fs::path acc;
            bool first = true;
            for (auto& part : _current_dir) {
                std::string s = part.string();
                if (s.empty()) continue;
                if (!first) { ImGui::SameLine(0,2); ImGui::TextUnformatted("/"); ImGui::SameLine(0,2); }
                first = false;
                acc /= part;
                fs::path nav = acc;
                ImGui::PushStyleColor(ImGuiCol_Button,        {0,0,0,0});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {.2f,.2f,.3f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {.5f,.75f,1.f,1.f});
                if (ImGui::SmallButton(s.c_str())) _navigate(nav.string());
                ImGui::PopStyleColor(3);
            }
        }

        if (!_error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, {1.f,.3f,.3f,1.f});
            ImGui::Text("Error: %s", _error.c_str());
            ImGui::PopStyleColor();
        }

        ImGui::Separator();

        // ── Two-column layout: left=places, right=files ───────────────────────
        float lw = 170.f;
        ImGui::BeginChild("##places", {lw, -(ImGui::GetFrameHeightWithSpacing()*2+8)}, true);

        ImGui::PushStyleColor(ImGuiCol_Text, {1.f,.7f,0.f,1.f});
        ImGui::TextUnformatted("RECENT"); ImGui::PopStyleColor();
        ImGui::Separator();
        // Copy to local vector so recents.push() during click can't invalidate iterator
        std::vector<std::string> recent_copy = recents.items();
        std::string pending_recent;
        for (int _ri = 0; _ri < (int)recent_copy.size(); ++_ri) {
            const std::string& r = recent_copy[_ri];
            fs::path rp(r);
            std::string lbl = rp.filename().string();
            if (lbl.empty()) lbl = r;
            if (lbl.size() > 18) lbl = lbl.substr(0, 16) + "..";
            ImGui::PushID(_ri);
            bool clicked = ImGui::Selectable(lbl.c_str());
            ImGui::PopID();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r.c_str());
            if (clicked) { pending_recent = r; break; }
        }
        if (!pending_recent.empty()) {
            fs::path rp(pending_recent);
            std::error_code ec;
            if (fs::is_directory(rp, ec)) {
                _navigate(pending_recent);
            } else if (fs::is_regular_file(rp, ec)) {
                _navigate(rp.parent_path().string());
                std::strncpy(_filename_buf, rp.filename().string().c_str(),
                             sizeof(_filename_buf)-1);
                if (_mode == Mode::Load) {
                    _result = pending_recent;
                    recents.push(pending_recent);
                    _open = false;
                    ImGui::EndChild(); ImGui::End(); return true;
                }
            }
        }

        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, {.5f,.75f,1.f,1.f});
        ImGui::TextUnformatted("PLACES"); ImGui::PopStyleColor();
        ImGui::Separator();
        auto place = [&](const char* lbl, const std::string& path) {
            std::error_code ec;
            if (fs::is_directory(path, ec))
                if (ImGui::Selectable(lbl)) _navigate(path);
        };
        std::string home = _home();
        place("~ Home",      home);
        place("  Desktop",   home + "/Desktop");
        place("  Documents", home + "/Documents");
        place("  Music",     home + "/Music");
        place("  Downloads", home + "/Downloads");
        place("/ Root",      "/");

        ImGui::EndChild();
        ImGui::SameLine();

        // ── File list ─────────────────────────────────────────────────────────
        ImGui::BeginChild("##files", {0, -(ImGui::GetFrameHeightWithSpacing()*2+8)}, true);

        // Header row
        ImGui::PushStyleColor(ImGuiCol_Text, {.4f,.4f,.5f,1.f});
        ImGui::Text("%-42s %10s", "Name", "Size");
        ImGui::PopStyleColor();
        ImGui::Separator();

        // We need to detect navigation intent BEFORE modifying _entries.
        // Use a flag: if navigate is needed, store the target and do it after EndChild.
        std::string nav_target;
        std::string confirm_path;

        for (auto& e : _entries) {
            bool selected = (e.name == std::string(_filename_buf));

            char label[512];
            if (e.is_dir)
                std::snprintf(label, sizeof(label), "[DIR] %s", e.name.c_str());
            else {
                char sz[16]; _fmt_size(e.size, sz);
                std::snprintf(label, sizeof(label), "      %-40s %s", e.name.c_str(), sz);
            }

            ImGui::PushStyleColor(ImGuiCol_Text,
                e.is_dir ? ImVec4(.4f,.8f,1.f,1.f) : ImVec4(.91f,.91f,.94f,1.f));
            ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(.2f,.3f,.4f,1.f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(.25f,.35f,.45f,1.f));

            bool clicked = ImGui::Selectable(label, selected,
                               ImGuiSelectableFlags_AllowDoubleClick |
                               ImGuiSelectableFlags_SpanAllColumns);

            ImGui::PopStyleColor(3);

            if (clicked) {
                if (e.is_dir) {
                    // Single click = highlight; double click = enter
                    if (ImGui::IsMouseDoubleClicked(0))
                        nav_target = e.path.string();
                } else {
                    // Single click = select filename
                    std::strncpy(_filename_buf, e.name.c_str(), sizeof(_filename_buf)-1);
                    // Double click in Load mode = confirm immediately
                    if (ImGui::IsMouseDoubleClicked(0) && _mode == Mode::Load)
                        confirm_path = (fs::path(_current_dir) / e.name).string();
                }
            }
        }

        ImGui::EndChild();

        // Apply deferred navigation (AFTER EndChild, safe from iterator invalidation)
        if (!nav_target.empty()) _navigate(nav_target);

        // Apply deferred confirmation
        if (!confirm_path.empty()) {
            _result = confirm_path;
            recents.push(_result);
            _open = false;
            ImGui::End();
            return true;
        }

        // ── Filename bar + filter toggle + buttons ────────────────────────────
        float btn_w = 95.f;
        float extra = _mode==Mode::Load ? btn_w+8 : 0;
        ImGui::SetNextItemWidth(std::max(50.f,
            ImGui::GetContentRegionAvail().x - btn_w*2 - extra - 20));
        ImGui::InputText("##fname", _filename_buf, sizeof(_filename_buf));

        // Extension hint / show-all toggle
        if (_mode == Mode::Load && !_extensions.empty()) {
            ImGui::SameLine(0, 6);
            ImGui::PushStyleColor(ImGuiCol_Button,
                _show_all ? ImVec4(.2f,.5f,.2f,.7f) : ImVec4(.1f,.1f,.15f,.7f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                _show_all ? ImVec4(.4f,1.f,.4f,1.f) : ImVec4(.5f,.5f,.6f,1.f));
            if (ImGui::Button(_show_all ? "All files" : "Filtered", {btn_w, 0})) {
                _show_all = !_show_all;
                _refresh();
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::SameLine();
        const char* ok_lbl = (_mode == Mode::Load) ? "Open" : "Save";
        ImGui::PushStyleColor(ImGuiCol_Button,        {.3f,.1f,.5f,.8f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {.4f,.15f,.65f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,          {.8f,.5f,1.f,1.f});
        if (ImGui::Button(ok_lbl, {btn_w, 0})) {
            if (_filename_buf[0] != '\0') {
                _result = (fs::path(_current_dir) / _filename_buf).string();
                if (_mode == Mode::Load) recents.push(_result);
                _open = false; dismissed = true;
            }
        }
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::Button("Cancel", {btn_w, 0})) {
            _result.clear(); _open = false; dismissed = true;
        }

        ImGui::End();
        return dismissed;
    }

    bool        is_open() const { return _open; }
    std::string result()  const { return _result; }

private:
    struct Entry {
        std::string name;
        fs::path    path;
        bool        is_dir = false;
        uintmax_t   size   = 0;
    };

    std::string              _title;
    Mode                     _mode     = Mode::Load;
    fs::path                 _current_dir;
    std::vector<std::string> _extensions;
    std::vector<Entry>       _entries;
    char                     _filename_buf[512] = {};
    bool                     _open      = false;
    bool                     _show_all  = false;
    bool                     _editing_path = false;
    std::string              _result;
    std::string              _error;

    static std::string _home() {
        const char* h = std::getenv("HOME");
        return h ? h : "/";
    }

    void _navigate(const std::string& path) {
        fs::path p(path);
        std::error_code ec;
        if (!fs::is_directory(p, ec)) {
            _error = "Not a directory: " + path;
            return;
        }
        _current_dir = fs::canonical(p, ec);
        if (ec) _current_dir = p;   // canonical may fail on some FS
        _error.clear();
        _refresh();
    }

    // Lower-case a copy of a string
    static std::string _lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return s;
    }

    bool _ext_matches(const fs::path& p) const {
        if (_show_all || _extensions.empty()) return true;
        std::string ext = _lower(p.extension().string());
        for (auto& e : _extensions)
            if (_lower(e) == ext) return true;
        return false;
    }

    void _refresh() {
        _entries.clear();
        _error.clear();
        std::error_code ec;

        // Test if we can open the directory at all
        auto it = fs::directory_iterator(_current_dir, ec);
        if (ec) { _error = "Cannot read directory: " + ec.message(); return; }

        for (; it != fs::end(it); it.increment(ec)) {
            if (ec) { ec.clear(); continue; }   // skip unreadable entries, reset ec
            std::error_code ec2;
            std::string name = it->path().filename().string();
            if (name.empty() || name[0] == '.') continue;
            bool is_dir = it->is_directory(ec2);
            if (ec2) { ec2.clear(); is_dir = false; } // treat unreadable as file

            if (!is_dir && !_ext_matches(it->path())) continue;

            uintmax_t sz = 0;
            if (!is_dir) { sz = fs::file_size(it->path(), ec2); ec2.clear(); }

            _entries.push_back({name, it->path(), is_dir, sz});
        }

        std::sort(_entries.begin(), _entries.end(), [](const Entry& a, const Entry& b) {
            if (a.is_dir != b.is_dir) return (int)a.is_dir > (int)b.is_dir;
            // Case-insensitive alphabetical
            std::string al = a.name, bl = b.name;
            std::transform(al.begin(),al.end(),al.begin(),[](unsigned char c){return std::tolower(c);});
            std::transform(bl.begin(),bl.end(),bl.begin(),[](unsigned char c){return std::tolower(c);});
            return al < bl;
        });
    }

    void _go_up() {
        auto parent = _current_dir.parent_path();
        if (parent != _current_dir) _navigate(parent.string());
    }

    static void _fmt_size(uintmax_t sz, char* buf) {
        if      (sz < 1024ULL)              std::snprintf(buf,16,"%llu B",  (unsigned long long)sz);
        else if (sz < 1024ULL*1024)         std::snprintf(buf,16,"%.1f KB", sz/1024.0);
        else if (sz < 1024ULL*1024*1024)    std::snprintf(buf,16,"%.1f MB", sz/(1024.0*1024));
        else                                std::snprintf(buf,16,"%.2f GB", sz/(1024.0*1024*1024));
    }
};
