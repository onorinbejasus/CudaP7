#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <assert.h>
#include "myserver.hh"
#include <fcntl.h>
#include <vector>

#include "networking.hh"
#include "Mesh.hh"
//#define PORT 2222
#define MAX_FD 10

void handle_clients_line(int i, char buf[]);

client_t clients[MAX_FD];

std::vector<struct m_triangle> mesh;
struct m_triangle *m_temp;

int main(int argc, char * argv[]){

	if(argc < 2){
		printf("Usage: ./main <port number>\n");
		exit(-1);
	}
		
	printf("Master Module Running, waiting for clients\n");

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
	
	Mesh *m = new Mesh();
	m->loadModelData("dragon.obj", &mesh);
		
	printf("mesh size: %i\n", mesh.size());
			
	fd_set fd_all;
	fd_set fd_read;

	//clear fd sets
	FD_ZERO(&fd_all);
	FD_ZERO(&fd_read);

	//add server
	FD_SET(server_fd, &fd_all);

	while(1){
		fd_read = fd_all;
		if(select(MAX_FD+1, &fd_read, NULL, NULL, NULL) == 0){
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

			float size[1600 * 3];
			
			for(int ii = 0; ii <1600 * 3; ii++)
				size[ii] = 55.100;
			
			send( clients[num_clients].fd, (float*)size, sizeof(float) * 1600 * 3, 0);
								
			
//			send(clients[num_clients].fd, (struct m_triangle*)test, sizeof(struct m_triangle), 0);
			
			num_clients++;
		}
		for(i=0; i<num_clients; i++){	
			if(FD_ISSET(clients[i].fd, &fd_read)){
				bzero(clients[i].inbuf, MAX_LINE);
				if((recv_length = recv(clients[i].fd, clients[i].inbuf, MAX_LINE+1, 0)) == 0){
					close(clients[i].fd);
					FD_CLR(clients[i].fd, &fd_all);
					memset(&clients[i], '\0', sizeof(client_t));
					printf("Client Removed\n");
					//num_clients--; commented due to dirty removal of clients
				}else{
					handle_clients_line(i, clients[i].inbuf);
				}
			}
		}
	}

	//end main
}

void handle_clients_line(int i, char buf[]){
	printf("%s", buf);
	send(clients[i].fd, buf, strlen(buf), 0);

}
