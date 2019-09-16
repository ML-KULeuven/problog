CC = gcc
BUILD_DIR = build
DEBUG_DIR = debug
BUILD_CFLAGS = -std=c99 -O2 -Wall -finline-functions -Iinclude -DNDEBUG
DEBUG_CFLAGS = -std=c99 -O0 -g -Wall -Iinclude
OS := $(shell uname -s)
LIBRARY_FLAGS := -Llib/${OS} -lsdd -lm

EXEC_FILE = sdd
SRC = src/main.c src/getopt.c \
	src/fnf/compiler.c src/fnf/fnf.c src/fnf/io.c src/fnf/utils.c 
HEADERS = include/sddapi.h include/compiler.h

OBJS = $(patsubst src/%.c,obj/%.o,$(SRC))
BUILD_OBJS = $(addprefix $(BUILD_DIR)/, $(OBJS))
BUILD_EXEC = $(BUILD_DIR)/$(EXEC_FILE)
DEBUG_OBJS = $(addprefix $(DEBUG_DIR)/, $(OBJS))
DEBUG_EXEC = $(DEBUG_DIR)/$(EXEC_FILE)

SRC_DIRS = $(shell find src/ -mindepth 1 -type d)
OBJ_DIRS = $(patsubst src/%,obj/%,$(SRC_DIRS))
BUILD_DIRS = $(addprefix $(BUILD_DIR)/, $(OBJ_DIRS))
DEBUG_DIRS = $(addprefix $(DEBUG_DIR)/, $(OBJ_DIRS))

$(BUILD_EXEC): $(BUILD_DIRS) $(BUILD_OBJS)
	$(CC) $(BUILD_OBJS) $(LIBRARY_FLAGS) -o $@

$(BUILD_DIRS):
	mkdir -p $(BUILD_DIRS)

$(BUILD_DIR)/obj/%.o: src/%.c $(HEADERS)
	$(CC) $(BUILD_CFLAGS) -c $< -o $@

.PHONY: debug
debug: $(DEBUG_EXEC)

$(DEBUG_EXEC): $(DEBUG_DIRS) $(DEBUG_OBJS)
	$(CC) $(DEBUG_OBJS) $(LIBRARY_FLAGS) -o $@

$(DEBUG_DIRS):
	mkdir -p $(DEBUG_DIRS)

$(DEBUG_DIR)/obj/%.o: src/%.c $(HEADERS)
	$(CC) $(DEBUG_CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(BUILD_OBJS) $(BUILD_EXEC) $(DEBUG_OBJS) $(DEBUG_EXEC)
