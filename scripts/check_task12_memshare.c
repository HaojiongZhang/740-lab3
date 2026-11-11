/* Run on the board beside libmemshare.so; validates real allocation/FD APIs. */
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); return 1; } } while (0)
int main(void) {
    void *h=dlopen("./libmemshare.so",RTLD_NOW|RTLD_LOCAL);
    CHECK(h);
    void *(*alloc)(size_t)=dlsym(h,"AllocateSharedBuffer");
    void (*release)(void*)=dlsym(h,"FreeSharedBuffer");
    int (*fd)(void*)=dlsym(h,"GetSharedBufferFd");
    CHECK(alloc && release && fd);
    CHECK(fd(NULL)==-1); release(NULL);
    CHECK(alloc(0)==NULL); CHECK(alloc(SIZE_MAX)==NULL);
    CHECK(alloc((size_t)2147483648ULL)==NULL);
    const size_t sizes[]={1,4095,4096,4097,393216,1572864,6291456};
    for (int round=0;round<3;round++) {
        void *buffers[7]; int descriptors[7];
        for (int i=0;i<7;i++) {
            buffers[i]=alloc(sizes[i]); CHECK(buffers[i]);
            CHECK((uintptr_t)buffers[i] % getpagesize()==0);
            descriptors[i]=fd(buffers[i]); CHECK(descriptors[i]>=0);
            CHECK(fcntl(descriptors[i],F_GETFD)>=0);
            CHECK(fd(buffers[i])==descriptors[i]);
            volatile unsigned char *b=buffers[i];
            b[0]=0x35; b[sizes[i]-1]=0x79; CHECK(b[sizes[i]-1]==0x79);
        }
        for (int i=6;i>=0;i--) release(buffers[i]);
        for (int i=0;i<7;i++) CHECK(fcntl(descriptors[i],F_GETFD)==-1);
    }
    void *ordinary=malloc(4096); CHECK(ordinary); CHECK(fd(ordinary)==-1); free(ordinary);
    CHECK(dlclose(h)==0);
    puts("PASS: 21 real shared allocations, page alignment, live FDs, FD cleanup, memory access, invalid sizes, NULL, ordinary-heap rejection");
    return 0;
}
