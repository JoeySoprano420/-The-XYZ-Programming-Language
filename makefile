CC = gcc
CFLAGS = -Wall -O2 -fPIC
AR = ar
ARFLAGS = rcs

TARGET_STATIC = libxyzrt.a
TARGET_SHARED = libxyzrt.so
SRC = libxyzrt.c
OBJ = $(SRC:.c=.o)
HDR = libxyzrt.h

all: $(TARGET_STATIC) $(TARGET_SHARED)

$(OBJ): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -c $(SRC) -o $(OBJ)

$(TARGET_STATIC): $(OBJ)
	$(AR) $(ARFLAGS) $@ $(OBJ)

$(TARGET_SHARED): $(OBJ)
	$(CC) -shared -o $@ $(OBJ) -lpthread

clean:
	rm -f $(OBJ) $(TARGET_STATIC) $(TARGET_SHARED)

install: all
	mkdir -p /usr/local/include /usr/local/lib
	cp $(HDR) /usr/local/include/
	cp $(TARGET_STATIC) /usr/local/lib/
	cp $(TARGET_SHARED) /usr/local/lib/
	ldconfig

.PHONY: all clean install
