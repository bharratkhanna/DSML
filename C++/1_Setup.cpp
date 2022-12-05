#include <bits/stdc++.h>
using namespace std;

int main(){
	char c = 'a';
	int a = 3;
	double b = 3.5;
	a = 4.5;
	bool d = true;
	cout<<c<<" "<<a<<" "<<b<<" "<<d<<endl;

	// long int, long long int
	long int e = 10000000;
	cout<<e<<endl;

	int i = 1;
	cout<<i++<<endl; // this should come as as 2 but came 1 so calculation after print
	cout<<++i<<endl; 

	// Relational Operators
	cout<<(a-1 == i)<<endl;

	// Characters
	cout<<(c+1)<<endl;
	cout<<int(c)<<endl;


}