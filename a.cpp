#include <iostream>
#include <string>

using namespace std;

void solve() {
    string s;
    cin >> s;
    int n = s.size();

    // 检查长度是否为偶数
    // 检查开头是否不是 ')'
    // 检查结尾是否不是 '('
    if (n % 2 == 0 && s[0] != ')' && s[n-1] != '(') {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}