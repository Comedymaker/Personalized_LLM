# Makefile for LLM Fine-tuning

# 配置
PYTHON := python
TRAIN_SCRIPT := run_trainer.py
EVAL_SCRIPT := evaluation/eval.py
TRAIN_LOG_FILE := results/output.txt
EVAL_LOG_FILE := results/output1.txt
REQUIREMENTS := requirements.txt
VENV := venv

.PHONY: all train install clean help

all: install train  # 默认执行完整流程

train:  # 运行训练并记录日志
	@echo "▶️ 开始训练! 日志保存到 ${TRAIN_LOG_FILE}"
	@${PYTHON} ${TRAIN_SCRIPT} > ${TRAIN_LOG_FILE} 2>&1
	@echo "✅ 训练完成! 查看日志: ${TRAIN_LOG_FILE}"

install:  # 安装Python依赖
	@echo "🔧 安装依赖..."
	@pip install -r ${REQUIREMENTS}

clean:  # 清理输出文件
	@echo "🧹 清理工作区..."
	@rm -f ${TRAIN_LOG_FILE}
	@rm -rf outputs/

help:  # 显示帮助信息
	@echo "可用命令:"
	@echo "  make install    安装依赖"
	@echo "  make train      启动训练并记录日志"
	@echo "  make clean      删除日志和输出文件"
	@echo "  make all        完整流程（安装+训练）"
	@echo "  make help       显示此帮助信息"

eval:
	@${PYTHON} ${EVAL_SCRIPT} > ${EVAL_LOG_FILE} 2>&1
