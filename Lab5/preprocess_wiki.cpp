#include <iostream>
#include <fstream>
#include <regex>
#include <cctype>

using namespace std;

bool is_empty_line(string line) {
    for (auto &c : line) {
        if (!isspace(c)) return false;
    }
    return true;
}

int main(void) {
    fstream fin("zhwiki_500.txt", ios::in), fout("zhwiki_500_processed.txt", ios::out);
    char line[100000];
    while (!fin.eof()) {
        fin.getline(line, 100000);
        
        string str = regex_replace(line, regex("<.*>"), ""); // 删除标签
        if (is_empty_line(str)) continue;                    // 空行跳过
        fout << line << endl;
    }
    fin.close();
    fout.close();
    return 0;
}