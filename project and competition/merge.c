#include<stdio.h>
int main(){
	int N,M,Q,x[100],s[100],y[100],t[100],a[100],b[100],l[100],z[100],u[100],slot[100],i,j,q=1,r,n=0,m=0,p,k,y1[100];
	scanf("%d %d %d",&N,&M,&Q);
	for(i=0;i<N;i++){
		scanf("%d",&x[i]);//distrance ac//dont know this gonna work with one line
	}
	for(i=0;i<N;i++){
		scanf("%d",&s[i]);//amount ac
	}
	for(i=0;i<M;i++){
		scanf("%d",&y[i]);//distrance dc
	}
	for(i=0;i<M;i++){
		scanf("%d",&t[i]);//amount dc
	}
	for(i=0;i<Q;i++){//error and number of slot that want to release
		scanf("%d %d %d",&a[i],&b[i],&l[i]);
	}
	for(k=0;k<Q;k++){
		for(j=0;j<M;j++){
		y1[j]=y[j]*a[k]+b[k];
	}
	
	//rearrange
	n=0;
	m=0;
	for(i=0;n<=N||m<=M;i++){
		if(x[n]<y1[m]){
			z[i]=x[n];
			u[i]=s[n];
			n+=1;
		}
		else if(x[n]==y1[m]){
			z[i]=x[n];
			u[i]=s[n]+t[m];
			n+=1;
			m+=1;
		}
		else{
			z[i]=y1[m];
			u[i]=t[m];
			m+=1;
		}
	}
	q=1;
	for(j=0;j<i;j++){
		for(p=0;p<u[j];p++){
			slot[q]=z[j];
			q++;
		}
	}
	for(r=1;r<q;r++){
		if(l[k]==r){
			printf("%d\n",slot[r]);
			break;
		}
	}
	}
}/*5 2 6
1
2
3
4
5
1
1
1
1
1
1
5
1
1
1 0 1
1 0 3
1 1 1
1 1 3
1 -2 1
1 -2 3*/
