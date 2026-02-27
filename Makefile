.PHONY: all build clean musl install-musl help

# 默认目标
all: build

# 标准构建
build:
	cargo build --release

# 静态链接构建（需要musl target）
musl:
	rustup target add x86_64-unknown-linux-musl
	cargo build --release --target x86_64-unknown-linux-musl

# 检查是否已安装musl工具链
check-musl:
	@which musl-gcc > /dev/null 2>&1 || \
		(echo "需要安装 musl-tools:" && \
		echo "  Arch: sudo pacman -S musl" && \
		echo "  Ubuntu/Debian: sudo apt install musl-tools musl-dev" && \
		echo "  Fedora: sudo dnf install musl-gcc" && \
		exit 1)

# 安装musl依赖（Arch Linux）
install-musl-arch:
	sudo pacman -S musl

# 安装musl依赖（Ubuntu/Debian）
install-musl-deb:
	sudo apt install musl-tools musl-dev

# 清理
clean:
	cargo clean

# 帮助
help:
	@echo "可用命令:"
	@echo "  make build          - 标准构建"
	@echo "  make musl           - 静态链接构建"
	@echo "  make install-musl-arch   - 安装musl工具链(Arch)"
	@echo "  make install-musl-deb    - 安装musl工具链(Debian/Ubuntu)"
	@echo "  make clean          - 清理构建文件"
