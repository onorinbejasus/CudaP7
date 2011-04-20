#ifndef _MASTER_H_
#define _MASTER_H_

#define MAX_LINE 128

typedef struct{
	int fd;
	char hostname[MAX_LINE];
	char inbuf[MAX_LINE];
	char outbuf[MAX_LINE];
} client_t;

#endif
