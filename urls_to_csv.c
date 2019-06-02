#include <stdio.h>
#include <stdlib.h>

int main(){

	FILE *inFile = fopen("original_urls.txt","r");
	FILE *outFile = fopen("urls.txt","w");
	char str[600];
	char id[200];
	char num[200];
	char url[200];
	//int count = 0;

	while(fgets(str,600,inFile)!=NULL){
		//printf("%s",str);	
		id[0] = '\0';
		num[0] = '\0';
		url[0] = '\0';
		sscanf(str,"%[^_]_%[^\t ]\t%[^\n\r]",id,num,url);
		fprintf(outFile,"%s,%s,%s\n",id,num,url);
		str[0] = '\0';
		//printf("%s\n",url);
		//count++;
		//printf("%d\n",count);
		//if(count > 14197126){
		//	printf("%s,%s,%s\n",id,num,url);
		//	break;
		//}
	}

	//while(fscanf(inFile,"n%d_%d %s",id,num,str)==1){
	//	printf("here\n");
	//	printf("%d,%d,%s",*id,*num,str);	
	//	break;
	//}
	fclose(inFile);
	fclose(outFile);

	return 0;
}
