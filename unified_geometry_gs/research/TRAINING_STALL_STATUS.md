# 训练后“思考卡住”的当前证据

2026-10-16：同一exec session7120已实际exit0，自动完成control/candidate两30k训练
和全部14后续提取/评价阶段（总16）。其GT采用门槛失败，结果完整保存，非卡在训练结束。
独立全量审计51212及只读posthoc34419也都exit0。没有这些任务仍在等待的证据；
勿把下方旧running快照当当前真值。新queue的自动推进确实生效，仍不能声称客户端/网络
回复流断开的未知根因已修复。

2026-10-15新记录：本轮开始实际WSL进程/GPU检查没有历史训练或编译活进程，GPU空闲；
原预检没有产物，才执行只读预检。该预检及四臂full-smoke随后均真正exit0。
新30k pair已在history_gap_pilot_v1运行，exec session7120，control子PID401曾现场确认Rl+，
JSONL/compact进度持续增加。controller会自动推进候选及全部固定质量评价，无需聊天触发。
PID可能随阶段更换，继续时必须查manifest与实际进程，不能凭本段旧快照重启。
这证明新队列在观测时正常，不证明客户端/网络回复中断已根治。

2026-10-13；新诊断不覆盖旧记录，不因聊天中断重启训练。

已确认两种不同情况：

1. 旧corrected_quality55_v1父队列PID405曾被主动SIGSTOP；控制子PID425完成后
   父队列未推进，旧manifest仍running。此前已经核验完成的控制模型、保存偏离记录，
   退出旧父队列并标superseded_before_candidate。不能恢复这个已淘汰队列。
   证据 `results/training_stall_diagnosis_v1.json`、原queue_hold与protocol_deviation记录。
2. 最新线程只读记录中，两轮明确failed，错误都是
   `stream disconnected before completion: An error occurred while processing your request`。
   见 `results/thread_stream_diagnosis_v2.json`，保留turn ID、时间和原始错误/request ID。
   另外一轮completed且无错误，当前轮inProgress。它们证实回复流曾断开；
   不能识别更底层网络/客户端/服务原因，也不能证明由训练结束触发。

本次实际WSL检查：factor_holdout55_v2健康v4、directional_field_world_v1健康v1
均为terminal_manifest，status complete，匹配进程为空。新报告脚本也exit0。
因此当前没有证据说明这些已结束任务仍在等待训练或CUDA发生死锁。
健康检查只是一时的进程/产物快照，不能保证未来连接不会中断。

后续执行约定已落实在研究runner：阶段命令顺序执行、保存PID/退出码与终态；
数值检查逐点原子保存结果与progress，异常保存failed和traceback。
终态文件和实际产物优先于聊天中是否已输出结论；超时先检查同一进程，不重复启动。
完整数据保存到文件，聊天只显示小型状态和实质发现，减少大量日志占用。
以上改进不能修复未知的回复流断开原因，也不应谎称已经彻底解决应用卡顿。

最新研究与可恢复入口见 `CONTINUATION.md`。
