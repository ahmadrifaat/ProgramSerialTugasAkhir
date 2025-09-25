CC = gcc

CFLAGS = -O3 -Wall -Wextra

LDFLAGS = -lm
PKG_LIBS = glew glfw3
PKGFLAGS = $(shell pkg-config --cflags --libs $(PKG_LIBS))

TARGET = program_serial
SRC = programSerial.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(PKGFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET) log_simulasi_serial.csv screenshot_serial_*.png eta_serial_*.asc
.PHONY: all clean
