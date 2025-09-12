//图的遍历 
/*#include<stdio.h>
#include<stdlib.h>
int visited[100]={0};
struct Edge{
	int value;
	struct Edge* next;
};
struct VNode{
	struct Edge* head;
	int data;
};
struct Graph{
	int vnum,edgenum;
	struct VNode* vnode;
};
void CreateGraph(struct Graph* g){
	scanf("%d%d",&g->vnum,&g->edgenum);
	g->vnode=(struct VNode*)malloc(g->vnum*sizeof(struct VNode));
	for(int i=0;i<g->vnum;i++){
		g->vnode[i].head=NULL;
		g->vnode[i].data=0;
	}
	int current,next;
	for(int i=0;i<g->edgenum;i++){
		scanf("%d %d",&current,&next);
		struct Edge* temp=(struct Edge*)malloc(sizeof(struct Edge)); 
		temp->value=next-1; 
		temp->next=g->vnode[current-1].head;
		g->vnode[current-1].head=temp;
	}
}
void DFS(struct Graph g,int start){
	if(visited[start])
	return ;
	visited[start]=1;
	struct Edge* head=g.vnode[start].head;
	printf("%d ",start+1);
	while(head){
		if(visited[head->value]==0){
		    DFS(g,head->value);
		}
		head=head->next;
	}
} 
int main(){
	struct Graph g;
	CreateGraph(&g);
	for(int i=0;i<g.vnum;i++) 
	DFS(g,i);
	return 0;
}*/
//快速排序
/*#include<stdio.h>
void swap(int* a,int* b){
	int temp=*a;
	*a=*b;
	*b=temp;
}
int partition(int *a,int left,int right){
	int pivot=a[left];
	int k=left;
	for(int i=k+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			swap(&a[i],&a[k]);
		}
	}
	swap(&a[left],&a[k]);
	return k;
}
void quick_sort(int* a,int left,int right){
	if(left>=right)
	return ;
	int k=partition(a,left,right); 
	quick_sort(a,left,k-1);
	quick_sort(a,k+1,right);
}

int main(){
	int n;
	scanf("%d",&n);
	int a[10000];
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	quick_sort(a,0,n-1);
	for(int i=0;i<n;i++)
	printf("%d ",a[i]);
	return 0;
} */

//归并排序
/*#include<stdio.h>
void  merge(int *a,int left,int mid,int right){
	int  temp[right-left+1];
	int k=0;
	int i=left;
	int j=mid;
	while(i<mid&&j<right){
		if(a[i]<=a[j]){
			temp[k++]=a[i++];
		}else
		temp[k++]=a[j++];
	}while(i<mid){
		temp[k++]=a[i++];
	}while(j<right){
		temp[k++]=a[j++];
	}
	for(int i=left;i<right;i++){
		a[i]=temp[i-left];
	}
}
void merge_sort(int *a,int left,int right){
	if(left>=right)
	return ;
	int mid=left+(right-left)/2;
	merge_sort(a,left,mid);
	merge_sort(a,mid+1,right);
	merge(a,left,mid,right);
}
int main(){
	int n;
	scanf("%d",&n);
	int a[10000];
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	merge_sort(a,0,n);
	for(int i=0;i<n;i++){
		printf("%d ",a[i]);
	}
	return 0;
}*/

//快速排序 
/*#include<stdio.h>
#include<time.h>
#include<stdlib.h>
int Partition(int* a,int left,int right){
	int ran,temp;
	 srand((unsigned int)(time(NULL))); 
	ran=left+rand()%(right-left+1);
	temp=a[ran];
	a[ran]=a[left];
	a[left]=temp;
	int pivot=a[left];
	int k=left+1;
	for(int i=left+1;i<=right;i++){
		if(a[i]<=pivot){
			temp=a[i];
			a[i]=a[k];
			a[k++]=temp;
		}
	}temp=a[k-1];
	a[k-1]=a[left];
	a[left]=temp;
	return k-1;
}
void Quick_sort(int* a,int left,int right){
	if(left>=right)
	return ;
	int k=Partition(a,left,right);
	Quick_sort(a,left,k-1);
	Quick_sort(a,k+1,right);
}
int main(){
	int n;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++)
	scanf("%d",&a[i]);
	Quick_sort(a,0,n-1);
	for(int i=0;i<n;i++)
	printf("%d ",a[i]);
	return 0;
}*/

//寻找最大最小，次大次小值 
/*#include <stdio.h>
#include <limits.h> 
struct result {
    int max;
    int secondMax;
    int min;
    int secondMin;
};
struct result findExtrema(int arr[], int left, int right) {
    struct result re;
    if (left == right) { // 只有一个元素
        re.max = arr[left];
        re.secondMax = INT_MIN;
        re.min = arr[left];
        re.secondMin = INT_MAX;
        return re;
    } else if (left + 1 == right) { // 只有二个元素
        if (arr[left] > arr[right]) {
            re.max = arr[left];
            re.secondMax = arr[right];
            re.min = arr[right];
            re.secondMin = arr[left];
        } else {
            re.max = arr[right];
            re.secondMax = arr[left];
            re.min = arr[left];
            re.secondMin = arr[right];
        }
        return re;
    } else { // 多于两个元素
        int mid = (left + right) / 2;
        struct result leftRes = findExtrema(arr, left, mid);
        struct result rightRes = findExtrema(arr, mid + 1, right);
        // 合并结果
        // 最大值和次最大值
        if (leftRes.max > rightRes.max) {
            re.max = leftRes.max;
            re.secondMax = (leftRes.secondMax > rightRes.max) ? leftRes.secondMax : rightRes.max;
        } else {
            re.max = rightRes.max;
            re.secondMax = (rightRes.secondMax > leftRes.max) ? rightRes.secondMax : leftRes.max;
        }
        
        // 最小值和次最小值
        if (leftRes.min < rightRes.min) {
            re.min = leftRes.min;
            re.secondMin = (leftRes.secondMin < rightRes.min) ? leftRes.secondMin : rightRes.min;
        } else {
            re.min = rightRes.min;
            re.secondMin = (rightRes.secondMin < leftRes.min) ? rightRes.secondMin : leftRes.min;
        }
        
        return re;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int a[n];
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }
    struct result re = findExtrema(a, 0, n - 1);
    printf("%d %d\n", re.max,re.secondMax);
    return 0;
}*/

//三分查找 
/*#include<stdio.h>
int search(int *a,int left,int right,int k){
	int mid1=left+(right-left)/3;
	int mid2=right-(right-left)/3;
	if(left>right)
	return -1;
    if(k==a[mid1])
	return mid1;
	else if(k==a[mid2])
	return mid2;
	if(k<a[mid1])
	return search(a,left,mid1-1,k);
	else if(k>a[mid1]&&k<a[mid2])
	return search(a,mid1+1,mid2-1,k);
	else if(k>a[mid2])
	return search(a,mid2+1,right,k);
}
int main(){
	int n,k;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++)
	scanf("%d",&a[i]);
	scanf("%d",&k);
	printf("%d",search(a,0,n-1,k));
	return 0;
}*/

//查找第k小元素 
/*#include<stdio.h>
#include<time.h>
#include<stdlib.h>
int Partition(int* a,int left,int right){
	int ran,temp;
	 srand((unsigned int)(time(NULL))); 
	ran=left+rand()%(right-left+1);
	temp=a[ran];
	a[ran]=a[left];
	a[left]=temp;
	int pivot=a[left];
	int k=left+1;
	for(int i=left+1;i<=right;i++){
		if(a[i]<=pivot){
			temp=a[i];
			a[i]=a[k];
			a[k++]=temp;
		}
	}temp=a[k-1];
	a[k-1]=a[left];
	a[left]=temp;
	return k-1;
}
int Quick_sort(int* a,int left,int right,int k1){
	int k=Partition(a,left,right);
	if(k1==k+1)
	return a[k];
	if(k1>k+1)
	return Quick_sort(a,k+1,right,k1);
	else
	return Quick_sort(a,left,k-1,k1);
	
}
void Insert_sort(int* a,int n){
	for(int i=1;i<n;i++){
		if(a[i]<a[i-1]){
			int temp=a[i];
			int j;
			for(j=i-1;j>=0;j--){
				if(a[j]>temp){
					a[j+1]=a[j];
				}else 
				break;
			}a[j+1]=temp;
		} 
	}
}
int main(){
	int n,k;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++)
	scanf("%d",&a[i]);
	scanf("%d",&k);
	printf("%d",Quick_sort(a,0,n-1,k));
	return 0;
}*/

//寻找最接近的元素 
/*#include<stdio.h>
#include<math.h>
int f(int* a,int k,int n){
	int left=0;
	int right=n-1;
	if(k>=a[right])
	return a[right];
	if(k<=a[left])
	return a[left];
	while(right-left>1){
		int mid=(right-left)/2+left;
		if(k<a[mid]){
			right=mid;
		}else if(k>a[mid]){
			left=mid;
		}else if(a[mid]==k){
			return k;
		}
	}if(fabs(k-a[left])<=fabs(k-a[right]))
	return a[left];
	return a[right];
}
int main(){
	int n,m;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++) scanf("%d",&a[i]);
	scanf("%d",&m);
	int k;
	while(m){
		scanf("%d",&k);
		int temp=f(a,k,n); 
		printf("%d\n",temp);
		m--;
	}
	return 0;
}*/

//背包问题 
/*#include<stdio.h>
#include<math.h>
struct Bag{
	int index;
	double weight;//单个物体总重量 
	double profit;//单个物体总收益 
	double p;//单位物体收益 
	double w;//每个物品要拿的重量 
};
int main(){
	int n;
	double c;
	double cnt=0;
	scanf("%d%lf",&n,&c);
	struct Bag b[n];
	for(int i=0;i<n;i++){
		scanf("%d%lf%lf",&b[i].index,&b[i].profit,&b[i].weight);
		b[i].p=b[i].profit*1.0/b[i].weight;
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1-i;j++){
			if(b[j].p<b[j+1].p){
				struct Bag temp=b[j];
				b[j]=b[j+1];
				b[j+1]=temp;
			}
		}
	}
	for(int i=0;i<n;i++){
		if(cnt+b[i].weight>c){
			b[i].w=c-cnt;
			b[i].w=b[i].w*1.0/b[i].weight;
			break;
		}b[i].w=1.0;
		cnt+=b[i].weight;
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1-i;j++){
			if(b[j].index>b[j+1].index){
				struct Bag temp=b[j];
				b[j]=b[j+1];
				b[j+1]=temp;
			}
		}
	}for(int i=0;i<n;i++){
		printf("%.2f ",b[i].w);
	}
	return 0;
} */

//带时限的作业问题
/*#include<stdio.h>
struct Job{
	int index;
	int profit;
	int deadline;
	int flag;
};
int main(){
	int n;
	scanf("%d",&n);
	struct Job j[n];
	for(int i=0;i<n;i++){
		scanf("%d%d%d",&j[i].index,&j[i].profit,&j[i].deadline);
		j[i].flag=0;
	}
	for(int i=0;i<n;i++){
		for(int k=0;k<n-1-i;k++){
			if((j[k].profit<j[k+1].profit)||((j[k].profit==j[k+1].profit)&&j[k].deadline>j[k+1].deadline)){
				struct Job temp=j[k];
				j[k]=j[k+1];
				j[k+1]=temp;
			}
		}
	}int time[n];
	for(int i=0; i<n; i++){
		time[i] = 0;
	}
	for(int i=0; i<n; i++){
		for(int k=j[i].deadline-1; k>=0; k--){
			if(time[k] ==0 ){
				time[k] = 1;
				j[i].flag=1;
				break;
			}
		}
	}
	for(int i=0;i<n;i++){
		for(int k=0;k<n-1-i;k++){
			if(j[k].index>j[k+1].index){
				struct Job temp=j[k];
				j[k]=j[k+1];
				j[k+1]=temp;
			}
		}
	}for(int i=0;i<n;i++){
		if(j[i].flag==1){
			printf("%d ",j[i].index);
		}
	}
	return 0;
} */

//乘船问题
/*#include<stdio.h>
struct People{
	int weight;
	int flag;
};
int main(){
	int n,totalweight;
	int cnt=0;
	scanf("%d",&n);
	struct People p[n];
	for(int i=0;i<n;i++){
		scanf("%d",&p[i].weight);
		p[i].flag=0;
	}scanf("%d",&totalweight);
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1-i;j++){
			if(p[j].weight>p[j+1].weight){
				struct People temp=p[j];
				p[j]=p[j+1];
				p[j+1]=temp;
			}
		}
	}int left,right;
	left=0;
	right=n-1;
	while(left<=right){
		if(p[left].weight+p[right].weight<=totalweight){
			cnt++;
			right--;
			left++;
		}else{
			cnt++;
			right--;
		}
	}
	printf("%d",cnt);
	return 0;
}*/

//取快递问题 
/*#include<stdio.h>
void swap(double* a,double* b){
	double temp=*a;
	*a=*b;
	*b=temp;
}
int partition(double *a,int left,int right){
	double pivot=a[left];
	int k=left;
	for(int i=k+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			swap(&a[i],&a[k]);
		}
	}
	swap(&a[left],&a[k]);
	return k;
}
void quick_sort(double* a,int left,int right){
	if(left>=right)
	return ;
	int k=partition(a,left,right); 
	quick_sort(a,left,k-1);
	quick_sort(a,k+1,right);
}
int main(){
	int n;
	int cnt=0;
	scanf("%d",&n);
	double w[n];
	for(int i=0;i<n;i++){
		scanf("%lf",&w[i]);
	}quick_sort(w,0,n-1);
	int left,right;
	left=0;
	right=n-1;
	while(left<=right){
		if(w[left]+w[right]<=3){
			cnt++;
			right--;
			left++;
		}else{
			cnt++;
			right--;
		}
	}
	printf("%d",cnt);
	return 0;
} */

//最长非递减子序列问题
/*#include<stdio.h>
void Calculate(int* a,int* dp,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<i;j++){
			if(a[j]<=a[i]){
				dp[i]=dp[i]>(dp[j]+1)?dp[i]:(dp[j]+1);
			}
		}
	}
}
int main(){
	int n,max,maxi;
	scanf("%d",&n);
	int a[n],dp[n];
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
		dp[i]=1;
	}
	Calculate(a,dp,n);
	max=dp[0];
	for(int i=1;i<n;i++){
		if(max<dp[i]){
			max=dp[i];
			maxi=i;
		}
	}
	printf("%d",dp[maxi]);
	return 0;
} */

//最长公共子序列
/*#include<stdio.h>
#include<string.h>
int dp[10000][10000]={0};
int Calculate(char* a,int n1,int n2,char* b){
	for(int i=1;i<=n1;i++){
		for(int j=1;j<=n2;j++){
			if(a[i]==b[j]){
				dp[i][j]=dp[i-1][j-1]+1;
			}else{
				dp[i][j]=dp[i-1][j]>dp[i][j-1]?dp[i-1][j]:dp[i][j-1];
			}
		}
	}
	return dp[n1][n2];
}
int main(){
	int n1,n2;
	char ch1[1000],ch2[1000],ch3[1000],ch4[1000];
	scanf("%s",ch1);
	scanf("%s",ch2);
	n1=strlen(ch1);
	n2=strlen(ch2);
	for(int i=0;i<n1;i++){
		ch3[i+1]=ch1[i];
	}for(int i=0;i<n2;i++){
		ch4[i+1]=ch2[i];

	}
	printf("%d",Calculate(ch3,n1,n2,ch4));
	return 0;
} */
//0/1背包问题
/*#include<stdio.h>
int dp[100][1000]={0};
int Calculate(int*w,int*p,int n,int c){
	for(int i=1;i<=n;i++){
		for(int j=c;j>=1;j--){
			dp[i][j]=dp[i-1][j];
			if(j-w[i]>=0){
				dp[i][j]=dp[i][j]>(dp[i-1][j-w[i]]+p[i])?dp[i][j]:(dp[i-1][j-w[i]]+p[i]);
			}
		}
	}
	return dp[n][c];
}
int main(){
	int n,c;
	scanf("%d",&n);
	scanf("%d",&c);
	int w[n+1],p[n+1]; 
	for(int i=1;i<n+1;i++) scanf("%d",&w[i]);
	for(int i=1;i<n+1;i++) scanf("%d",&p[i]);  
	printf("%d",Calculate(w,p,n,c));
	return 0;
}*/


//矩阵连乘问题
/*#include <stdio.h>
#include <limits.h>
int dp[1000][1000];
int Calculate(int* p, int n) {
    for (int i = 0; i < n - 1; i++) {
        dp[i][i] = 0;
    }
    for (int r = 2; r <= n - 1; r++) { 
        for (int i = 0; i <= n - r - 1; i++) { 
            int j = i + r - 1; 
            dp[i][j] = INT_MAX; 
            for (int k = i; k < j; k++) { 
                int cost = dp[i][k] + dp[k+1][j] + p[i] * p[k+1] * p[j+1];
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                }
            }
        }
    }
    return dp[0][n-2]; 
}
int main() {
    int n;
    scanf("%d", &n);
    int a[n + 1];
    for (int i = 0; i < n + 1; i++) { 
        scanf("%d", &a[i]);
    }

    printf("%d", Calculate(a, n + 1));
    return 0;
}
*/

//多段图问题
/*#include<stdio.h>
#include<stdlib.h>
int dp[1000]={0};
struct Edge{
	int adjvertex;
	int value;
	struct Edge* next;
};
struct Vertex{
	struct Edge* head;
};
struct Graph{
	int vertexnum;
	int edgenum;
	struct Vertex* v;
};
void CreateGraph(struct Graph* g){
	int tempv,tempe,tempvalue;
	scanf("%d%d",&g->vertexnum,&g->edgenum);
	g->v=(struct Vertex*)malloc(g->vertexnum*sizeof(struct Vertex));
	for(int i=0;i<g->vertexnum;i++){
		g->v[i].head=NULL;
	}
	for(int i=0;i<g->edgenum;i++){
		scanf("%d%d%d",&tempv,&tempe,&tempvalue);
		struct Edge* temp=(struct Edge*)malloc(sizeof(struct Edge));
		temp->adjvertex=tempe;
		temp->value=tempvalue;
		temp->next=g->v[tempv].head;
		g->v[tempv].head=temp;
	}
}
int Calculate(struct Graph* g)
int main(){
	struct Graph g;
	CreateGraph(&g);
	return 0;
} */

//游艇问题 
/*#include<stdio.h>
#include<limits.h>
int dp[201]={0};
int Calculate(int n,int a[201][201]){
	for(int i=1;i<=n;i++){
	    dp[i]=INT_MAX;
	    dp[1]=0;
		for(int j=1;j<i;j++){
			dp[i]=dp[i]<(dp[j]+a[j][i])?dp[i]:(dp[j]+a[j][i]);
		}
	}
	return dp[n];
}
int main(){
    int n;
	scanf("%d",&n);
	int p[201][201]={0};
	for(int i=1;i<=n;i++){
		for(int j=i+1;j<=n;j++){
			scanf("%d",&p[i][j]);
		}
	}	
	printf("%d",Calculate(n,p));
	return 0;
}*/

//走方格 
/*#include<stdio.h>
int dp[31][31];
int Calculate(int n,int m){
	if(n==1||m==1) dp[n][m]=1;
	if(dp[n][m]==-1){
		if(m%2==0&&n%2==0)
		dp[n][m]=0;
		else
		dp[n][m]=Calculate(n-1,m)+Calculate(n,m-1);
		return dp[n][m];
	}else
	return dp[n][m];
}
int main(){
	int n,m;
	scanf("%d%d",&n,&m);
	for(int i=0;i<=n;i++){
		for(int j=0;j<=m;j++)
		dp[i][j]=-1;
	}
	printf("%d",Calculate(n,m));
	return 0;
}*/

//过河卒 
/*#include<stdio.h>
long long dp[100][100];
int Check(int currentx,int currenty,int x,int y){
	if(currentx==x&&currenty==y)
	return -1;
	else if(currentx+2==x&&currenty-1==y)
	return -1;
	else if(currentx-2==x&&currenty-1==y)
	return -1;
	else if(currentx+2==x&&currenty+1==y)
	return -1;
	else if(currentx-2==x&&currenty+1==y)
	return -1;
	else if(currentx+1==x&&currenty-2==y)
	return -1;
	else if(currentx-1==x&&currenty-2==y)
	return -1;
	else if(currentx+1==x&&currenty+2==y)
	return -1;
	else if(currentx-1==x&&currenty+2==y)
	return -1;
	return 0;
}
long long Calculate(int m,int n,int x,int y){
	if(m<0||n<0)
	return 0;
	if(dp[m][n]==-1){
		if(Check(m,n,x,y)==-1)
		dp[m][n]=0;
		else{
			dp[m][n]=Calculate(m-1,n,x,y)+Calculate(m,n-1,x,y);
		}
		return dp[m][n];
	}else 
	return dp[m][n];
}
int main(){
	int m,n;
	int hoursex,hoursey;
	scanf("%d%d%d%d",&m,&n,&hoursex,&hoursey);
	for(int i=0;i<=m;i++){
		for(int j=0;j<=n;j++)
		dp[i][j]=-1;
	}dp[0][0]=1;
	printf("%lld",Calculate(m,n,hoursex,hoursey));
	return 0;
}*/

//最大连续子序列
/*#include<stdio.h>
int dp[1000]={0};
int Calculate(int* a,int n){
	for(int i=1;i<=n;i++){
		dp[i]=(dp[i-1]+a[i])>a[i]?(dp[i-1]+a[i]):a[i];
	}int max=dp[1];
	for(int i=1;i<=n;i++){
		if(max<dp[i])
		max=dp[i];
	}return max;
}
int main(){
	int n;
	scanf("%d",&n);
	int a[n+1];
	for(int i=1;i<=n;i++){
		scanf("%d",&a[i]);
	}printf("%d",Calculate(a,n));
	return 0;
} */
 
//装箱问题
/*#include<stdio.h>
int dp[31][20001];
int Calculate(int v,int n,int* volume){
	for(int i=0;i<=v;i++)
	dp[0][i]=v;
	for(int i=1;i<=n;i++){
		for(int j=v;j>=0;j--){
			dp[i][j]=dp[i-1][j];
			if(j-volume[i]>=0){
				dp[i][j]=dp[i][j]<(dp[i-1][j-volume[i]]-volume[i])?dp[i][j]:(dp[i-1][j-volume[i]]-volume[i]);
			}
			
		}
	}return dp[n][v];
}
int main(){
    int v,n;
	scanf("%d%d",&v,&n);
	int volume[n+1];
	for(int i=1;i<=n;i++) scanf("%d",&volume[i]);
	printf("%d",Calculate(v,n,volume));	
	return 0;
} */

//开心的金明
/*#include<stdio.h>
long long dp[25][30000]={0};
long long Calculate(int n,int m,int* p,int* im){
	for(int i=1;i<=m;i++){
		for(int j=n;j>=0;j--){
			dp[i][j]=dp[i-1][j];
			if(j-p[i]>=0)
			dp[i][j]=dp[i][j]>(dp[i-1][j-p[i]]+p[i]*im[i])?dp[i][j]:(dp[i-1][j-p[i]]+p[i]*im[i]);
		}
	}return dp[m][n];
}
int main(){
	int n,m;
	scanf("%d%d",&n,&m);
	int price[m+1],importance[m+1];
	for(int i=1;i<=m;i++){
		scanf("%d%d",&price[i],&importance[i]);
	}
	printf("%lld",Calculate(n,m,price,importance));
	return 0;
} */

//红牌 
/*#include<stdio.h>
#include<limits.h> 
long long dp[2001][2001];
long long Calculate(int n,int m,int time[][2001]){
	for(int i=1;i<=m;i++)
	dp[i][1]=time[i][1];
	for(int j=2;j<=n;j++){
		for(int i=2;i<=m;i++){
			dp[i][j]=dp[i][j-1]+time[i][j];
			dp[i][j]=dp[i][j]<(dp[i-1][j-1]+time[i][j])?dp[i][j]:(dp[i-1][j-1]+time[i][j]);
		}dp[1][j]=(dp[1][j-1]+time[1][j])<(dp[m][j-1]+time[1][j])?(dp[1][j-1]+time[1][j]):(dp[m][j-1]+time[1][j]);
	}int min=INT_MAX;
	for(int i=1;i<=m;i++)
	if(min>dp[i][n])
	min=dp[i][n];
	return min;
}
int main(){
	int n,m;
	scanf("%d%d",&n,&m);
	int time[2001][2001];
	for(int i=1;i<=m;i++)
	for(int j=1;j<=n;j++)
	scanf("%d",&time[i][j]);
	printf("%lld",Calculate(n,m,time));
	return 0;
} */

//小A点菜
/*#include<stdio.h>
int dp[1000][1000];
int Calculate(int n,int m,int* a){
	dp[0][0]=1;
	for(int i=1;i<=m;i++){
		for(int j=0;j<=n;j++){
			if(j-a[i]>=0){
				dp[i][j]=dp[i-1][j]+dp[i-1][j-a[i]];
			}else
			dp[i][j]=dp[i-1][j];
		}
	} 
	return dp[m][n];
}
int main(){
    int n,m;
	scanf("%d%d",&m,&n);
	int a[m+1];
	for(int i=1;i<=m;i++) scanf("%d",&a[i]);
	printf("%d",Calculate(n,m,a));
	return 0;
} */

//n皇后问题 
/*#include<stdio.h>
#include<math.h>
int k[14][14];
int x[14];
int cnt=0;
int Check(char ch[14][14],int start,int n){
	if(k[start][x[start]]==-1)
	return -1;
	for(int i=0;i<start;i++){
		if(x[i]==x[start]||(abs(start-i)==abs(x[start]-x[i])))
		return -1;
	}return 0;
}
void DFS(char ch[14][14],int start,int n){
	for(int i=0;i<n;i++){
		x[start]=i;
		if(Check(ch,start,n)==0){
			DFS(ch,start+1,n);
			if(start==n-1){
				cnt++;
				return ;
			}
		}x[start]=0;
	}
}
int main(){
	int n;
	scanf("%d",&n);
	char ch[14][14];
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			scanf(" %c",&ch[i][j]);
			if(ch[i][j]=='.'){
				k[i][j]=-1;
			}else{
				k[i][j]=0;
			}
		}
	}
	DFS(ch,0,n);
	printf("%d",cnt);
	return 0;
}*/

//n皇后问题-非递归 
/*#include<stdio.h>
#include<math.h>
int k[14][14];
int x[14]={-1};
int Check(char ch[14][14],int start,int n){
	if(k[start][x[start]]==-1)
	return -1;
	for(int i=0;i<start;i++){
		if(x[i]==x[start]||abs(start-i)==abs(x[start]-x[i]))
		return -1;
	}return 0;
}
int main(){
	int n;
	int total=0;
	scanf("%d",&n);
	char ch[14][14];
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			scanf(" %c",&ch[i][j]);
			if(ch[i][j]=='.'){
				k[i][j]=-1;
			}else{
				k[i][j]=0;
			}
		}
	}
	int cnt=0;
	while(cnt>=0){
		x[cnt]++;
		while(x[cnt]<n&&Check(ch,cnt,n)==-1) x[cnt]++;
		if(x[cnt]>=n){
			x[cnt]=-1;
			cnt--;
		}else{
			if(cnt==n-1){
				total++;
			}else{
				cnt++;
			}
		}
	}
	printf("%d",total);
	return 0;
}*/

//子集和数 
/*#include<stdio.h>
int x[1000]={0};
int Partition(int left,int right,int*a){
	int pivot=a[left];
	int k=left;
	for(int i=left+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			int temp=a[k];
			a[k]=a[i];
			a[i]=temp;
		}
	}int temp=a[k];
	a[k]=a[left];
	a[left]=temp;
	return k;
}
void Quick_Sort(int left,int right,int* a){
	if(left>=right)
	return ;
	int k=Partition(left,right,a);
	Quick_Sort(left,k-1,a);
	Quick_Sort(k+1,right,a);
}
void DFS(int n,int m,int k,int* a,int* cnt,int s,int r){
	if(s==m){
		(*cnt)++;
		for(int i=0;i<n;i++)
		printf("%d ",x[i]);
		printf("\n");
		return ;
	}
	if(s+a[k]<=m&&k<n){
		x[k]=1;
		DFS(n,m,k+1,a,cnt,s+a[k],r-a[k]);
		x[k]=0;
	}if(s+a[k+1]<=m&&s+r-a[k]>=m){
		x[k]=0;
		DFS(n,m,k+1,a,cnt,s,r-a[k]);
	}
}
int main(){
	int n,m;
	int s=0;
	int r=0;
	scanf("%d%d",&n,&m);
	int a[n];
	for(int i=0;i<n;i++) scanf("%d",&a[i]);
	int cnt=0;
	Quick_Sort(0,n-1,a);
	for(int i=0;i<n;i++)
		r+=a[i];
	DFS(n,m,0,a,&cnt,s,r);
	if(cnt==0)
	printf("no solution!");
	return 0;
}*/

//N-Queen-非对称解 
/*#include<stdio.h>
#include<math.h>
int Check(int k,int n,int* x){
	for(int i=0;i<k;i++){
		if(abs(i-k)==abs(x[i]-x[k])||x[i]==x[k])
		return -1;
	}return 1;
}
void DFS(int k,int n,int* x,int* cnt){
	if(k==n){
		(*cnt)++;
		if(x[0]<=(n-1)/2) {
		for(int i=0;i<n;i++)
			printf("%d ",x[i]);
			printf("\n");
		}
			return ;
	}
	for(int i=0;i<n;i++){
		x[k]=i;
		if(Check(k,n,x)==1){
			DFS(k+1,n,x,cnt);
		}x[k]=0;
	}
}
int main(){
	int n,cnt;
	cnt=0;
	scanf("%d",&n);
	int x[n];
	for(int i=0;i<n;i++) x[i]=0;
	DFS(0,n,x,&cnt);
	return 0;
}*/

//数字三角形
/*#include<stdio.h>
int dp[1001][1001];
int a[1001][1001];
int Calculate(int a[1001][1001],int n){
	dp[1][1]=a[1][1];
	for(int i=2;i<=n;i++){
		for(int j=1;j<=i;j++){
			dp[i][j]=dp[i-1][j-1]+a[i][j];
			dp[i][j]=dp[i][j]>(dp[i-1][j]+a[i][j])?dp[i][j]:(dp[i-1][j]+a[i][j]);
		}
	}int max=dp[n][1];
	for(int i=2;i<=n;i++)
	if(max<dp[n][i])
	max=dp[n][i];
	return max;
}
int main(){
	int n;
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		for(int j=1;j<=i;j++){
			scanf("%d",&a[i][j]);
		}
	}
	printf("%d",Calculate(a,n));
	return 0;
}*/

//n色图问题
/*#include<stdio.h>
int x[101]={0};
int Check(int current,int k,int G[101][101],int n){
	for(int i=0;i<current;i++){
		if(G[current][i]==1&&x[current]==x[i])
		return -1;
	}
	return 1;
}
void DFS(int G[101][101],int* cnt,int n,int k,int current){
	if(current==n){
		(*cnt)++;
		for(int i=0;i<n;i++)
		printf("%d ",x[i]);
		printf("\n");
		return ;
	}
	for(int i=1;i<=k;i++){
		x[current]=i;
		if(Check(current,k,G,n)==1){
			DFS(G,cnt,n,k,current+1);
		}
	}
}
int main(){
	int n,m,k;
	int cnt=0;
	int G[101][101]={0};
	scanf("%d%d%d",&n,&m,&k);
	for(int i=0;i<m;i++){
		int u,v;
		scanf("%d%d",&u,&v);
		G[u][v]=1;
		G[v][u]=1;
	}
	DFS(G,&cnt,n,k,0);
	if(cnt==0)
	printf("No solution");
	return 0;
}*/

//哈密顿环
/*#include<stdio.h>
int g[100][100];
int x[100];
int Check(int current,int n){
	if(current>0&&g[x[current]][x[current-1]]==0)
	return -1;
	if(current==n-1&&g[x[current]][x[0]]==0)
	return -1;
	for(int i=0;i<current;i++){
		if(x[i]==x[current])
		return -1;
	}return 1;
}
void DFS(int n,int current,int* cnt){
	if(current==n){
		(*cnt)++;
		if((*cnt)==1){
			for(int i=0;i<n;i++)
			printf("%d ",x[i]);
			printf("%d",x[0]);
		}
		return ;
	}
	for(int i=0;i<n;i++){
		x[current]=i;
		if(Check(current,n)==1){
			DFS(n,current+1,cnt);
		}
	}
}
int main(){
	int n,m;
	int cnt=0;
	scanf("%d%d",&n,&m);
	for(int i=0;i<m;i++){
		int u,v;
		scanf("%d%d",&u,&v);
		g[u][v]=1;
		g[v][u]=1;
	}
	DFS(n,1,&cnt);
	if(cnt==0)
	printf("No solution");
	return 0;
}*/

//组合数问题
/*#include<stdio.h>
int x[100];
int Check(int current,int n,int k){
	for(int i=1;i<current;i++){
		if(x[i]==x[current]||x[current]<=x[i]){
			return -1;
		}
	}return 1;
}
void DFS(int current,int n,int k,int k1){
	if(k==0){
		for(int i=1;i<=k1;i++)
		printf("%d ",x[i]);
		printf("\n");
	}
	for(int i=1;i<=n;i++){
		x[current]=i;
		if(Check(current,n,k)==1){
			DFS(current+1,n,k-1,k1);
		}
	}
}
int main(){
	int n,k;
	scanf("%d%d",&n,&k);
	DFS(1,n,k,k);
	return 0; 
}*/

//最大正方形 
/*#include<stdio.h>
int dp[101][101];
int a[101][101];
int Calculate(int n,int m){
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			if(i-1>=0&&j-1>=0&&a[i][j]!=0){
				int min=dp[i-1][j]<dp[i-1][j-1]?dp[i-1][j]:dp[i-1][j-1];
				min=min<dp[i][j-1]?min:dp[i][j-1];
				dp[i][j]=min+1;
			}
		}
	}
	int max=dp[0][0];
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			if(max<dp[i][j])
			max=dp[i][j];
		}
	}return max;
}
int main(){
	int m,n;
	scanf("%d%d",&n,&m);
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			scanf("%d",&a[i][j]);
			if(a[i][j]==1)
			dp[i][j]=1;
		}
	}
	printf("%d",Calculate(n,m));
	return 0;
} */

//最短下降路径 
/*#include<stdio.h>
int dp[102][102];
int Calculate(int n,int a[101][101]){
	for(int i=0;i<n;i++) dp[0][i]=a[0][i];
	for(int i=1;i<n;i++){
		for(int j=0;j<n;j++){
			int min=dp[i-1][j];
			if(j==0){
				min=min<dp[i-1][j+1]?min:dp[i-1][j+1];
			}else if(j==n-1){
				min=min<dp[i-1][j-1]?min:dp[i-1][j-1];
			}else{
				min=min<dp[i-1][j-1]?min:dp[i-1][j-1];
				min=min<dp[i-1][j+1]?min:dp[i-1][j+1];
			}dp[i][j]=min+a[i][j];
		}
	}int min=dp[n-1][0];
	for(int i=1;i<n;i++)
	min=min<dp[n-1][i]?min:dp[n-1][i];
	return min;
}
int main(){
	int n;
	scanf("%d",&n);
	int a[101][101];
	for(int i=0;i<n;i++)
	for(int j=0;j<n;j++)
	scanf("%d",&a[i][j]);                                                                                    
	printf("%d\n",Calculate(n,a));
	return 0;
} */

//15迷问题 
/*#include<stdio.h>
#include<string.h>
int a[5][5];
int less[16];
int Check(int *temp,int x,int y){
	for(int i=0;i<16;i++){
		for(int j=i+1;j<16;j++){
			if(temp[j]<temp[i])
			less[i]++;
		}
	}int cnt=x+y+2;
	for(int i=0;i<16;i++){ 
		cnt+=less[i];
	} 
	if(cnt%2==1)
	return -1;
	return cnt;
}
int main(){
	int n=1;
	int x,y;
	char ch[10];
	int temp[16];
	int cnt=0;
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				scanf("%s",ch);
				if(strcmp(ch,"#")==0)
				a[i][j]=16;
				else{
					for(int k=0;k<strlen(ch);k++){
						a[i][j]*=10;
						a[i][j]+=ch[k]-'0';
					}
				}
				temp[cnt++]=a[i][j];
				if(a[i][j]==16){
					x=i;
				    y=j;
				}
			}
		}
		if(Check(temp,x,y)==-1)
		printf("FALSE");
		else{
			printf("TRUE");
		}
	return 0;
} */

//是否存在路径-BFS 
/*#include<stdio.h>
#include<stdlib.h>
int visited[200000]={0};
struct Edge{
	int value;
	struct Edge* next;
};
struct VNode{
	struct Edge* head;
	int data;
};
struct Graph{
	int vnum,edgenum;
	struct VNode* vnode;
};

int q[200000];
void CreateGraph(struct Graph* g){
	scanf("%d%d",&g->vnum,&g->edgenum);
	g->vnode=(struct VNode*)malloc(g->vnum*sizeof(struct VNode));
	for(int i=0;i<g->vnum;i++){
		g->vnode[i].head=NULL;
		g->vnode[i].data=0;
	}
	int current,next;
	for(int i=0;i<g->edgenum;i++){
		scanf("%d %d",&current,&next);
		struct Edge* temp=(struct Edge*)malloc(sizeof(struct Edge)); 
		temp->value=next; 
		temp->next=g->vnode[current].head;
		g->vnode[current].head=temp;
		temp = (struct Edge*)malloc(sizeof(struct Edge));
        temp->value = current;
        temp->next = g->vnode[next].head;
        g->vnode[next].head = temp;
	}
}
int BFS(int start,int final,struct Graph g){
	int front=0;
	int rear=0;
	q[front++]=start;
	visited[start]=1;
	while(rear<front){
		int current=q[rear++];
		if(current==final)
		return 1;
		struct Edge* temp=g.vnode[current].head;
		while(temp!=NULL){
			if(visited[temp->value]==0){
				q[front++]=temp->value;
				visited[temp->value]=1;
			}temp=temp->next;
		}
	}return -1;
}
int main(){
	struct Graph g;
	CreateGraph(&g);
	int start,final;
	scanf("%d%d",&start,&final);
	if(BFS(start,final,g)==1){
		printf("true");
	}else{
		printf("false");
	}
	return 0;
}*/
//是否存在路径-DFS 
/*#include<stdio.h>
int a[200000][200000];
int DFS(int n,int m,int current,int final){
	if(current==final)
	return 1;
	for(int j=0;j<n;j++){
		if(a[current][j]==1){
			a[current][j]=0;
			if(DFS(n,m,j,final))
			return 1;
		}
	}return 0;
}
int main(){
	int n,m;
	int start,final;
	scanf("%d%d",&n,&m);
	for(int i=0;i<m;i++){
		int v,u;
		scanf("%d%d",&v,&u);
		a[v][u]=a[u][v]=1;
	}scanf("%d%d",&start,&final);
	if(DFS(m,n,start,final)==1){
		printf("true");
	}else{
		printf("false");
	}
	return 0;
}*/

//01迷宫
/*#include<stdio.h>
#include<string.h>
int ans[1000][1000];
char a[1000][1000];
int visited[1000][1000];
int kx[4]={0,0,1,-1};
int ky[4]={-1,1,0,0};
struct Queue{
	int dx,dy;
};
int rear;
int front;
struct Queue q[100004];
int Check(int currentx,int currenty,int cx,int cy,int n){
	if(visited[cx][cy]==1)
	return -1;
	else if(cx<0||cy<0||cx>=n||cy>=n)
	return -1;
	else if(a[currentx][currenty]=='0'&&a[cx][cy]=='0')
	return -1;
	else if(a[currentx][currenty]=='1'&&a[cx][cy]=='1')
	return -1;
	else
	return 1;
}
int BFS(int x,int y,int n){
	front=0;
	rear=0;
	int cnt=1;
	q[rear].dx=x;
	q[rear].dy=y;
	rear++;
	visited[x][y]=1;
	while(front<rear){
		int currentx=q[front].dx;
		int currenty=q[front].dy;
		for(int i=0;i<4;i++){
			int tempx,tempy;
		    tempx=currentx+kx[i];
		    tempy=currenty+ky[i];
		    if(Check(currentx,currenty,tempx,tempy,n)==1){
		    	q[rear].dx=tempx;
		    	q[rear].dy=tempy;
		    	visited[tempx][tempy]=1;
		    	rear++;
		    	cnt++;
			}
		}front++;
	}
	for(int i=0;i<rear;i++){
        ans[q[i].dx][q[i].dy]=cnt;
        visited[q[i].dx][q[i].dy]=0;
    } 
	return cnt;
}
int main(){
	int n,m;
	scanf("%d%d",&n,&m);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			scanf(" %c",&a[i][j]);
			ans[i][j]=-1;
		}
	}
	while(m--){
		int x,y;
		scanf("%d%d",&x,&y);
		if(ans[x-1][y-1]==-1)
		printf("%d\n",BFS(x-1,y-1,n));
		else
		printf("%d\n",ans[x-1][y-1]);
	}
	return 0;
} */

//学生分组
/*#include<stdio.h> 
int Calculate(int* a,int min,int max,int n){
	int temp1=0;
	int temp2=0;
	for(int i=0;i<n;i++){
		if(a[i]>max){
			temp1+=a[i]-max;
		}else if(a[i]<min){
			temp2+=min-a[i];
		}
	} 
	return temp1>temp2?temp1:temp2;
}
int Partition(int *a,int left,int right){
	int pivot=left;
	int k=left;
	for(int i=left+1;i<=right;i++){
		if(a[i]<a[pivot]){
			k++;
			int temp=a[i];
			a[i]=a[k];
			a[k]=temp;
		}
	}
	int temp=a[k];
	a[k]=a[left];
	a[left]=temp;
	return k;
}
void Quick_Sort(int *a,int left,int right){
	if(left>=right)
	return ;
	int k=Partition(a,left,right);
	Quick_Sort(a,left,k-1);
	Quick_Sort(a,k+1,right);
}
int main(){
	int n;
	scanf("%d",&n);
	int min,max;
	int cnt=0;
	int a[n];
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
		cnt+=a[i];
	}
	scanf("%d%d",&min,&max);
	if(cnt<min*n||cnt>max*n){
		printf("-1");
	}else{
		Quick_Sort(a,0,n-1);
	    printf("%d",Calculate(a,min,max,n));
	}
	return 0;
}*/

//最大连续区域 
/*#include<stdio.h>
int a[1000][1000];
int visited[1000][1000];
int max;
int dx[4]={0,0,1,-1};
int dy[4]={1,-1,0,0};
void DFS(int x,int y,int n,int m,int* cnt){
	if(a[x][y]==0||visited[x][y]==1)
	return ;
	(*cnt)++;
	max=max>*cnt?max:*cnt;
	for(int i=0;i<4;i++){
		int cx=dx[i]+x;
		int cy=dy[i]+y;
		visited[x][y]=1;
		if(cx>=0&&cx<m&&cy>=0&&cy<n){
			DFS(cx,cy,n,m,cnt);
		}
	}
}
int main(){
	int n,m;
	scanf("%d%d",&m,&n);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			scanf("%d",&a[i][j]);
		}
	}for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			if(a[i][j]==1){
				int cnt=0;
				DFS(i,j,n,m,&cnt);
			}
		}
	}
	printf("%d",max);
	return 0;
}*/

//NASA的食物计划 --二维背包问题dp 
/*#include<stdio.h>
int dp[400][400];
int Calculate(int* v,int* m,int* t,int n,int maxv,int maxm){
	for(int i=1;i<=n;i++){
		for(int j=maxv;j>=0;j--){
			for(int k=maxm;k>=0;k--){
				if(j-v[i]>=0&&k-m[i]>=0){
					dp[j][k]=(dp[j-v[i]][k-m[i]]+t[i])>dp[j][k]?(dp[j-v[i]][k-m[i]]+t[i]):dp[j][k];
				}
			}
		}
	}
	return dp[maxv][maxm];
}
int main(){
	int maxv,maxm;
	scanf("%d%d",&maxv,&maxm);
	int n;
	scanf("%d",&n);
	int v[51],m[51],k[51];
	for(int i=1;i<=n;i++) scanf("%d %d %d",&v[i],&m[i],&k[i]);
	
	printf("%d",Calculate(v,m,k,n,maxv,maxm));
	return 0;
}*/

//数字划分 
/*#include<stdio.h>
int Partition(int *a,int left,int right){
	int pivot=a[left];
	int k=left;
	for(int i=left+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			int temp=a[i];
			a[i]=a[k];
			a[k]=temp;
		}
	}
	int temp=a[left];
	a[left]=a[k];
	a[k]=temp;
	return k;
}
void Sort(int* a,int left,int right,int k){
	if(left==right)
	return ;
	int location=Partition(a,left,right);
	if(location==k)
	return ;
	else if(location<k){
		Sort(a,location+1,right,k);
	}else{
		Sort(a,left,location-1,k);
	}
}
int main(){
	int n;
	int s1=0,s2=0;
	scanf("%d",&n);
	int a[n+1];
	for(int i=1;i<=n;i++)
	scanf("%d",&a[i]);
	int mid=n/2;
    Sort(a,1,n,mid);
    for(int i=1;i<=mid;i++)
    s1+=a[i];
    for(int i=mid+1;i<=n;i++)
    s2+=a[i];
	printf("%d",s2-s1); 
	return 0;
} */

//最大与次大元素
/*#include<stdio.h>
#include<limits.h>
void Find(int *a, int left, int right, int *max, int *secmax) {
    if (left == right) {
        *max = a[left];
        *secmax=INT_MIN; 
        return;
    }
    
    int mid=(left+right)/2;
    int max1,secmax1,max2,secmax2;
    
    Find(a,left,mid,&max1,&secmax1);
    Find(a,mid+1,right,&max2,&secmax2);
    
    if (max1>max2) {
        *max=max1;
        *secmax=(secmax1>max2)?secmax1:max2;
    } else {
        *max=max2;
        *secmax=(secmax2>max1)?secmax2:max1;
    }
}
int main(){
	int n;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++) scanf("%d",&a[i]);
	int max,second;
	Find(a,0,n-1,&max,&second);
	printf("%d %d",max,second);
	return 0;
} */

//集合划分 
/*#include<stdio.h>
int Partition(int *a,int left,int right){
	int pivot=a[left];
	int k=left;
	for(int i=left+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			int temp=a[i];
			a[i]=a[k];
			a[k]=temp;
		}
	}
	int temp=a[left];
	a[left]=a[k];
	a[k]=temp;
	return k;
}
void Sort(int* a,int left,int right,int k){
	if(left==right)
	return ;
	int location=Partition(a,left,right);
	if(location==k)
	return ;
	else if(location<k){
		Sort(a,location+1,right,k);
	}else{
		Sort(a,left,location-1,k);
	}
}
int main(){
	int n;
	int s1=0,s2=0;
	scanf("%d",&n);
	int a[n+1];
	for(int i=1;i<=n;i++)
	scanf("%d",&a[i]);
	int mid=n/2;
    Sort(a,1,n,mid);
    for(int i=1;i<=mid;i++)
    s1+=a[i];
    for(int i=mid+1;i<=n;i++)
    s2+=a[i];
	printf("%d",s2-s1); 
	return 0;
} */

//二分查找最接近值 

/*#include<stdio.h>
int Search(int* a,int n,int temp){
	int left=1,right=n,mid;
	while(left<=right){
		mid=(left+right)/2;
		if(a[mid]<temp){
			left=mid+1;
		}else{
			right=mid-1;
		}
	}if(a[left]>=temp)
	return a[left];
	return -1;
}
void swap(int* a,int* b){
	int temp=*a;
	*a=*b;
	*b=temp;
}
int partition(int *a,int left,int right){
	int pivot=a[left];
	int k=left;
	for(int i=k+1;i<=right;i++){
		if(a[i]<=pivot){
			k++;
			swap(&a[i],&a[k]);
		}
	}
	swap(&a[left],&a[k]);
	return k;
}
void quick_sort(int* a,int left,int right){
	if(left>=right)
	return ;
	int k=partition(a,left,right); 
	quick_sort(a,left,k-1);
	quick_sort(a,k+1,right);
}
int main(){
	int n,k;
	scanf("%d%d",&n,&k);
	int a[n+1];
	for(int i=1;i<=n;i++) scanf("%d",&a[i]);
	 quick_sort(a,1,n);
	for(int i=0;i<k;i++){
		int temp;
		scanf("%d",&temp);
		printf("%d\n",Search(a,n,temp));
	}
	return 0;
} */


//解一元三次方程 
/*#include<stdio.h>
#include<math.h>
double f(double x){
	return 5.0*x*x*x-2.0*x*x+8.0; 
}
double Search(double x1,double x2){
	double mid;
	while(x2-x1>2*(1e-5)){
		mid=(x1+x2)/2.0;
		if(f(mid)*f(x1)<0){
			x2=mid;
		}else{
			x1=mid;
		}
	}return (x1+x2)/2.0;
}
int main(){
	double x1,x2;
	scanf("%lf %lf",&x1,&x2);
	printf("%.5f",Search(x1,x2));
	return 0;
}*/

//包裹运输 
/*#include<stdio.h>
int Check(int* a,int n,int time,int k){
	int sum=0;
	int t=1;
	for(int i=0;i<n;i++){
		if(a[i]>k)
		return 0;
		if(a[i]+sum<=k){
			sum+=a[i];
		}else{
			sum=a[i];
			t++;
		}
	}
	if(t>time)
	return 0;
	return 1;
}
int Search(int left,int right,int* a,int n,int time){
	int mid;
	while(left<=right){
		mid=left+(right-left)/2;
		if(Check(a,n,time,mid)==0){
			left=mid+1;
		}else{
			right=mid-1;
		}
	}return left;
}
int main(){
	int n;
	int sum=0;
	int time;
	int max=0;
	scanf("%d",&n);
	int a[n];
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
		sum+=a[i];
		if(max<a[i])
		max=a[i];
	}
	scanf("%d",&time);
	printf("%d",Search((sum/n)>max?(sum/n):max,sum,a,n,time));
	return 0;
}*/

//找零钱 
/*#include<stdio.h>
int main() {
    double n;
    scanf("%lf", &n);
    int n1 = (int)(n * 10); 
    int n2 = n1 / 10;        
    int n3 = n1 % 10;        
    int a1 = 0, a2 = 0, a5 = 0, a10 = 0, a20 = 0, a50 = 0, a100 = 0;
    int b1 = 0, b2 = 0, b5 = 0;
    a100 = n2 / 100;
    n2 %= 100;
    a50 = n2 / 50;
    n2 %= 50;
    a20 = n2 / 20;
    n2 %= 20;
    a10 = n2 / 10;
    n2 %= 10;
    a5 = n2 / 5;
    n2 %= 5;
    a2 = n2 / 2;
    n2 %= 2;
    a1 = n2; 
    b5 = n3 / 5;
    n3 %= 5;
    b2 = n3 / 2;
    n3 %= 2;
    b1 = n3;
    printf("0.1 %d\n0.2 %d\n0.5 %d\n1 %d\n2 %d\n5 %d\n10 %d\n20 %d\n50 %d\n100 %d\n",
           b1, b2, b5, a1, a2, a5, a10, a20, a50, a100);
    
    return 0;
}*/

//活动安排 
/*#include<stdio.h>
struct Job{
	int start;
	int end;
};
int main(){
	int n;
	int cnt=1;
	scanf("%d",&n);
	struct Job job[n];
	for(int i=0;i<n;i++){
		scanf("%d%d",&job[i].start,&job[i].end);
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1-i;j++){
			if(job[j].end>job[j+1].end){
				struct Job temp=job[j];
				job[j]=job[j+1];
				job[j+1]=temp;
			}
		}
	}
	int time=job[0].end;
	for(int i=1;i<n;i++){
		if(time<=job[i].start){
			cnt++;
			time=job[i].end;
		}
	}
	printf("%d",cnt);
}*/

//加油站 
/*#include<stdio.h>
int main(){
	int n,k;
	int a[10000];
	while(scanf("%d%d",&n,&k)==2){
		int flag=0;
		for(int i=0;i<=k;i++){
			scanf("%d",&a[i]);
			if(a[i]>n){
				flag=1;
			}
		}if(flag){
			printf("No Solution\n");
		}else{
			int sum=a[0];
			int cnt=0;
			for(int i=1;i<=k;i++){
				if(sum+a[i]<=n){
					sum+=a[i];
				}else{
					sum=a[i];
					cnt++;
				}
			}
			printf("%d\n",cnt);
		}
	}
	return 0;
}*/

//风险值 
/*#include<stdio.h>
#include<stdlib.h>
struct T{
	long long weight;
	long long strongth;
};
int Partition(struct T *a,int left,int right){
	struct T pivot=a[left];
	int k=left;
	for(int i=left+1;i<=right;i++){
		if(a[i].strongth+a[i].weight>=pivot.strongth+pivot.weight){
			k++;
			struct T temp=a[i];
			a[i]=a[k];
			a[k]=temp;
		}
	}
	struct T temp=a[left];
	a[left]=a[k];
	a[k]=temp;
	return k;
}
void Sort(struct T* a,int left,int right){
	if(left>=right)
	return ;
	int location=Partition(a,left,right);
	Sort(a,location+1,right);
	Sort(a,left,location-1);
}
int main(){
	int n;
	scanf("%d",&n);
	struct T a[n];
	for(int i=0;i<n;i++){
		scanf("%lld%lld",&a[i].weight,&a[i].strongth);
	}
	Sort(a,0,n-1);
	long long max=-a[n-1].strongth;
	long long w=a[n-1].weight;
	for(int i=n-2;i>=0;i--){
		long long temp=w-a[i].strongth;
		w+=a[i].weight;
		if(temp>max)
		max=temp;
	}
	printf("%lld",max);
	return 0;
}*/

//区间分组 
/*#include<stdio.h>
struct T{
	long long left;
	long long right;
};
int main(){
	int n;
	scanf("%d",&n);
	struct T a[n];
	for(int i=0;i<n;i++){
		scanf("%lld%lld",&a[i].left,&a[i].right);
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1-i;j++){
			if(a[j].left>a[j+1].left||(a[j].left==a[j+1].left&&a[j].right>a[j+1].right)){
				struct T temp=a[j];
				a[j]=a[j+1];
				a[j+1]=temp;
			}
		}
	}
	int flag=1;
	struct T current1,current2;
	current1=a[0];
	current2.left=-2;
	current2.right=-1;
	for(int i=1;i<n;i++){
		if(current1.right<a[i].left){
			current1=a[i];
		}else if(a[i].left<=current1.right&&current1.right<=a[i].right){
			if(a[i].left>current2.right){
				current2=a[i];
			}else if(a[i].left<=current2.right&&current2.right<=a[i].right){
				flag=0;
				break;
			}
		}
	}if(flag){
		printf("YES");
	}else{
		printf("NO");
	}
	return 0;
} */

//合并果子 
/*#include <iostream>
#include <queue>
#include <vector>
using namespace std;
int main() {
	int n;
	cin >> n;
	priority_queue<int, vector<int>, greater<int>> a;
	for (int i = 0; i < n; i++) {
		int num;
		cin >> num;
		a.push(num);
	}
	long long sum = 0;
	while (a.size() > 1) {
		int temp1 = a.top();
		a.pop();
		int temp2 = a.top();
		a.pop();
		int cost=temp1+temp2;
		sum+=cost;
		a.push(cost);
	}
	cout <<sum<< endl;
	
	return 0;
}*/


