#include <cstdio>
#include <cctype>
#include <iostream>
#include <regex>

using namespace std;

bool is_empty_line(string line) {
    for (char &c : line) {
        if (!isspace(c)) return false;
    }
    return true;
}

int main() {
    char str[] = "hahahaha<int>:shoot";
    string line = regex_replace(str, regex("<.*>"), "");
    char s[800]{};
    cout << line << endl;
    cout << is_empty_line(s) << endl;
}