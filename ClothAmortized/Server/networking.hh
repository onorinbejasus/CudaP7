#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include <unistd.h> //for size_t

/**
 * Connects to a listening server
 * Returns socket file descriptor
 */
int serv_connect(char* server_ip, int server_port);

/**
 * Sets a FD to be non blocking
 */
void set_nonblocking(int newclientfd);

/**
 * Accepts a connection with the client.
 * Returns client's socket file descriptor
 */
int accept_client(int sock);

/**
 * Initializes a TCP socket to listen
 * Returns server's socket file descriptor
 */
int setup_socket(int port);

/**
 * Closes a socket
 * Returns < 0 on error
 */
int close_socket(int sockfd);

/**
 * Reads a line to a socket
 */
ssize_t readline(int sockd, void *vptr, size_t maxlen);

/**
 * Writes a line to a socket
 */
ssize_t writeline(int sockd, const void *vptr, size_t n);

#endif /*_NETWORKING_H*/
