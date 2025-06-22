#include<bits/stdc++.h>
using namespace std;
string check(string input){
    string result;
    int note[26];
    for(int i=0;i<26;i++){
        note[i]='0';
    }
    for(size_t i=0;i<input.length();i++){
        if(note[input[i]-97]=='1'){//already found
            result+=input[i];
        }
        else{
            note[input[i]-97]='1';
            input.erase(i,1);
        }
    }
    if(input.empty()){
        return result;
    } 
    return check(result);
}
int main(){
    string input;
    getline(cin,input);
    cout<<check(input);
}
