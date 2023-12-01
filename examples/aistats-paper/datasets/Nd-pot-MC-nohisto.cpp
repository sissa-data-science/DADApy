/*
 Nd-pot-MC-nohisto.cpp
 */

//g++ Nd-pot-MC-nohisto.cpp -lm -lgsl -lgslcblas -O2 -o Nd-pot-MC-nohisto.out


#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
//#include "/u/sbp/mcarli/Dropbox/PhD/Codes/Libraries/my-rng.cpp"
#include "/home/matteo/Dropbox/Lavoro/Research/Codes/Libraries/my-rng.cpp"

using namespace std;


#define NDIM 6


double Prob_2d(double x, double y){

	return pow( 2.*exp(-(-1.5 + x)*(-1.5 + x) - (-2.5 + y)*(-2.5 + y)) + 3*exp(-2*x*x - 0.25*y*y) , 3 );
}


double Prob_2d_plus_harm(vector<double> v){	
	//harmonic potential = 6(x_i)^2
	double value=1;

	for (int i = 2; i < NDIM; ++i)
		value *= exp(-6*v[i]*v[i]);
	return value*=pow( 2.*exp(-(-1.5 + v[0])*(-1.5 + v[0]) - (-2.5 + v[1])*(-2.5 + v[1])) + 3*exp(-2*v[0]*v[0] - 0.25*v[1]*v[1]) , 3 );
	//f(x,y)=(2.*exp(-(-1.5 + x)*(-1.5 + x) - (-2.5 + y)*(-2.5 + y)) + 3*exp(-2*x*x - 0.25*y*y))**3
}


double Bias(double x){
	if ( x < -0.818048 || x > 2.33717 )
		return 0;
	else
		return log(1.02561*exp((3. - 5.*x)*x) + 0.117835*exp((6. - 4.*x)*x) +  0.00958554*exp((9. - 3.*x)*x) + 55.2596*exp(-6.*x*x));
}


double Prob(double x){
	return 1.02561*exp((3. - 5.*x)*x) + 0.117835*exp((6. - 4.*x)*x) + 0.00958554*exp((9. - 3.*x)*x) + 55.2596*exp(-6.*x*x);
}




int main()
{
//	int nt = 10000000000;
//	long int nt = 1200000000;
	int multipstride = 120000;
	long int stride = 1000;
	int exponent = 3;
//	int rejectcount = 0;
	vector<double> newpos(NDIM,0);
	vector<double> oldpos(NDIM,0.5);
	double deltaq = 0.5;
	double doubletemp;
	ofstream outf,outf1;
	string filename;

	RandomNumbers* rng = new RandomNumbers();

	cout << "AAAAAAAAAAAAAAAAAAAAAAAA cambia esponente stride" << endl;
	cout << "Starting MC simulation of length " << 1.2 << "E" << exponent+5 << " and stride 1E" << exponent << endl;



	//***************************//
 	// ALL OVER AGAIN BUT UNBIASED //
 	//***************************//


 	cout << "Running unbiased simulation" << endl;
 	outf.open("OUTPUT-1.2E8-stride1E3/u-time-series.dat");
	for (int i = 0; i < multipstride; ++i){
		if (i%100==0)
			cout << "Step " << i << "E3/1.2E8" << flush << "\r";
//		cout << "Step " << i << flush << "\r";
		for (int j = 0; j < stride; ++j){
			for (int d = 0; d < NDIM; ++d)
				newpos[d] = oldpos[d] + deltaq*(rng->u()-0.5);
			doubletemp = Prob_2d_plus_harm(newpos)/Prob_2d_plus_harm(oldpos);
			if (doubletemp > 1)
				for (int d = 0; d < NDIM; ++d)
					oldpos[d] = newpos[d];
			else if (doubletemp > rng->u())
				for (int d = 0; d < NDIM; ++d)
					oldpos[d] = newpos[d];
		}
	//		else
	//			++ rejectcount ;
		outf << i << "E" << exponent;
		for (int d = 0; d < NDIM; ++d)
			outf << " " << oldpos[d];
		outf << flush << endl;
	}
	outf.close();
//	cout << double(rejectcount)/nt;


	return 0;
}
