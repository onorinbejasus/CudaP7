#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "open_gl.hh"
#include "networking.hh"

#include <cuda_runtime.h>

#include "window.hh"
#include "setup.hh"
#include <iostream>
#include <cmath>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <assert.h>
#include <fcntl.h>
#include <vector>

#define PI 3.14159265
#define movingSpeed 10.0

#define MAX_LINE 4800

typedef struct{
	int fd;
	char hostname[MAX_LINE];
	char inbuf[MAX_LINE];
	char outbuf[MAX_LINE];
} client_t;

#define MAX_FD 10
client_t clients[MAX_FD];

int numCloths = 1;

bool dsim = false;
bool wind = false;
extern void step_func();
extern float *get_dataPtr();
extern uint *get_indexPtr();
extern uint numTriangles;

extern int size;

void handle_clients_line(int client){
	
	float data[size * 3] ;//= get_dataPtr();
	step_func();
	
	for(int ii = 0; ii < size * 3; ii++){
		data[ii] = 55;
	}
	
	send( clients[client].fd, (float*)data, sizeof(float) * size * 3, 0);	
	
}

int main(int argc, char **argv) {
   
	if(argc < 2){
		printf("Usage: ./main <port number>\n");
		exit(-1);
	}

	int i;
	int iter_fd;
	struct sockaddr_in clientaddr;
	struct hostent *h;
	unsigned int clen = sizeof(clientaddr);
	int recv_length = 0;
	int num_clients = 0;

	int port = atoi(argv[1]);
	int server_fd = setup_socket(port);	
	if(server_fd < 0){
		printf("Error Setting Up Server Socket, Exiting\n");
		exit(EXIT_FAILURE);
	}

	fd_set fd_all;
	fd_set fd_read;

	//clear fd sets
	FD_ZERO(&fd_all);
	FD_ZERO(&fd_read);

	//add server
	FD_SET(server_fd, &fd_all);
	
	initCuda(argc, argv);	
//	initGL(argc, argv);

	while(1){
				
		fd_read = fd_all;
	
		if(select(MAX_FD+1, &fd_read, NULL, NULL,  NULL) == 0){
			//no jobs needed
			continue;
		}
		if(FD_ISSET(server_fd, &fd_read)){
			//new connection request
			printf("New client found, attempting connect...\n");
			iter_fd = accept(server_fd, (struct sockaddr*)&clientaddr, &clen);
	
			if(iter_fd < 0){
				printf("Error connecting to client, rejecting...\n");
				continue;
			}
			printf("Connection Accepted\n");
	
			set_nonblocking(iter_fd);
			FD_SET(iter_fd, &fd_all);
	
			clients[num_clients].fd = iter_fd;
			printf("Client Connected on fd %d\n", iter_fd);
	
			uint indices[numTriangles * 3]; //= get_indexPtr();
			
			for(int ii = 0; ii < numTriangles * 3; ii++){
				indices[ii] = 66;
			}
			
			send( clients[num_clients].fd, (uint*)indices, sizeof(uint) * numTriangles * 3, 0);		
								
			num_clients++;
		}
		for(i=0; i<num_clients; i++){				
			
			if(FD_ISSET(clients[i].fd, &fd_read)){
				if((recv_length = recv(clients[i].fd, clients[i].inbuf, MAX_LINE+1, 0)) == 0){
					close(clients[i].fd);
					FD_CLR(clients[i].fd, &fd_all);
					memset(&clients[i], '\0', sizeof(client_t));
					printf("Client Removed\n");
				
				}else{
					handle_clients_line(i);
				
				}
			}
		}
	}
				
	return 0;
}


