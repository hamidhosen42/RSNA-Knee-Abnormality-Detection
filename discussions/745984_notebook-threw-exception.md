# Notebook Threw Exception

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745984
- **Author:** **Serendipity** (@stringyixuxi, Contributor) · _topic author_ · 2026-10-05 16:36 UTC · LB rank 1951
- **Topic votes:** 0 · **Comments/replies:** 10 · **Pinned:** no
- **Last comment:** 2026-10-13 00:34 UTC
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745984, raw JSON `_raw/745984.json`)

---

连续多次提交测评出现Notebook Threw Exception问题是因为什么呢，在notebook中运行是没有问题的，交给ai修改后还是出现相同的问题，该如何解决

---

## Comments (10, threaded, oldest first)

### [1] zhuo wamg

**zhuo wamg** (@zhuowamg, Contributor) · 2026-10-06 06:32 UTC · ▲1 · LB rank 364

I’ve had the same thing happen to me too, and it’s happened many times.

#### ↳ [1.1] Serendipity

**Serendipity** (@stringyixuxi, Contributor) · _topic author_ · 2026-10-07 03:14 UTC · LB rank 1951

你是怎么解决这种问题的呢，求指教

##### ↳ ↳ [1.1.1] zhuo wamg

**zhuo wamg** (@zhuowamg, Contributor) · 2026-10-08 02:07 UTC · LB rank 364

我现在都没有解决掉这个问题，昨天五次提交我有四次都评分失败了

### [2] Mingquan Zhuang

**Mingquan Zhuang** (@mingquanzhuang, Contributor) · 2026-10-08 10:23 UTC · LB rank 3190

先看失败的那次提交日志里第一条 traceback，而不是只看编辑器中交互运行的结果。提交会从保存的版本在干净环境重跑；草稿会话里已有的变量或临时文件，不能当作提交时一定存在。

我在这个比赛复现 Notebook 时会依次查：① Input 是否附上比赛数据和所有模型权重；比赛数据可能挂在 /kaggle/input/competitions/rsna-knee-abnormality-detection，也可能是 /kaggle/input/rsna-knee-abnormality-detection，先用 Path.exists() 确认；② 保存版本后从头 Run All 是否成功；③ 日志是否显示 GPU/内存不足或依赖安装失败；④ 最终 submission 文件是否按要求生成。

如果方便，贴出失败版本日志中最早的异常类型和附近几行（隐去个人信息/密钥），才能判断这次到底是哪一步失败。单凭 ‘Notebook Threw Exception’ 无法确定原因。

#### ↳ [2.1] Serendipity

**Serendipity** (@stringyixuxi, Contributor) · _topic author_ · 2026-10-08 13:18 UTC · LB rank 1951

run all是正常可以的logs中也是successfully，只是在提交运行后一两小时的时候直接Threw Exception，在保存运行是成功的

##### ↳ ↳ [2.1.1] Mingquan Zhuang

**Mingquan Zhuang** (@mingquanzhuang, Contributor) · 2026-10-09 02:05 UTC · LB rank 3190

明白了：你说的是保存版本的 Run All 和 logs 都显示成功，但提交任务过一两小时才失败。这时要把两次运行分开查。请在比赛的 Submissions 页面打开失败的那条记录，核对它绑定的 Notebook 版本/运行编号是否就是成功保存的版本，再看那条失败提交自己的日志（尤其末尾和第一条 traceback），不要只看保存版本页面的 “Successfully”。如果失败记录没有可见异常，建议保留失败提交链接、版本号、提交时间和脱敏后的日志/截图，发给比赛组织者或 Kaggle Support。单凭 “Notebook Threw Exception” 还无法判断是代码、资源限制还是评测侧问题；查清前也可以先避免连续消耗每日提交次数。

##### ↳ ↳ [2.1.2] Serendipity

**Serendipity** (@stringyixuxi, Contributor) · _topic author_ · 2026-10-09 03:28 UTC · LB rank 1951

好的感谢您，我已经给主办方发了邮件有后续我会发在评论区

##### ↳ ↳ [2.1.3] zhuo wamg

**zhuo wamg** (@zhuowamg, Contributor) · 2026-10-11 07:26 UTC · LB rank 364

我已经解决了这个问题，告诉你的codex or claude 具体情况和评分失败，然后让他们自己去对照之前评分成功的版本进行修改，经过这样我的修复版终于成功了

### [3] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-10-09 05:22 UTC · ▲-2

Thank you for Exception @stringyixuxi

### [4] (unknown)

**(unknown)** · 2026-10-13 00:34 UTC

_(no content — deleted or empty)_
