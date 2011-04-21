#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <vector>
#include "networking.hh"

#define MAXLINE 1000

/* int main(int argc, char *argv[]){
	if(argc < 3){
        printf("Need ip address and port number\n");
        exit(EXIT_FAILURE);
    }
		
	int port = atoi(argv[2]);
	char ip[20];
	strcpy(ip, argv[1]);
	
	int sock = serv_connect(ip, port);
	
	if(sock < 0){
		printf("Error Connecting to Server, Exiting\n");
		exit(EXIT_FAILURE);
	}
	
	float size[1600 * 3];
	readline(sock, (float*)size, sizeof(float) * 1600 * 3);
	
	for(int ii = 0; ii < 1600 * 3; ii++)
		printf("%f\n", size[ii]);
	
//	readline(sock, (struct m_triangle*)test, sizeof(struct m_triangle) );
					
//	for(int ii = 0; ii < 100; ii++)
//	 	printf("vert: %f %f %f\n", mesh[ii].vertices[0][0], mesh[ii].vertices[0][1], mesh[ii].vertices[0][2]);

	// char buf[MAXLINE];
	// 	sprintf(buf, "I am a test client\n");
	// 	writeline(sock, buf, strlen(buf));
	// 	bzero(buf, MAXLINE);
	// 	readline(sock, buf, MAXLINE);
	// 	printf("Received: %s\n", buf);
	// 	printf("Closing Connection\n");
	// 	close_socket(sock);

	exit(EXIT_SUCCESS);
}*/
