# 论文复现批量实验脚本
# 用于批量运行核心实验，验证论文结论

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "论文复现实验 - 批量运行脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否在项目根目录
if (-not (Test-Path "main CMomentum.py")) {
    Write-Host "错误: 请在项目根目录运行此脚本！" -ForegroundColor Red
    exit 1
}

# 检查数据集目录
if (-not (Test-Path "dataset")) {
    Write-Host "警告: dataset目录不存在，请先下载数据集！" -ForegroundColor Yellow
}

# 创建record目录（如果不存在）
if (-not (Test-Path "record")) {
    New-Item -ItemType Directory -Path "record" | Out-Null
    Write-Host "已创建record目录" -ForegroundColor Green
}

# 定义实验配置
$experiments = @(
    # 实验组1：Non-IID + 静态标签翻转（核心实验）
    @{
        Name = "Mean + Non-IID + Label Flipping"
        Aggregation = "mean"
        Attack = "label_flipping"
        Partition = "noniid"
        Priority = "HIGH"
    },
    @{
        Name = "Trimmed-Mean + Non-IID + Label Flipping"
        Aggregation = "trimmed-mean"
        Attack = "label_flipping"
        Partition = "noniid"
        Priority = "HIGH"
    },
    @{
        Name = "FABA + Non-IID + Label Flipping"
        Aggregation = "faba"
        Attack = "label_flipping"
        Partition = "noniid"
        Priority = "HIGH"
    },
    @{
        Name = "CC + Non-IID + Label Flipping"
        Aggregation = "cc"
        Attack = "label_flipping"
        Partition = "noniid"
        Priority = "HIGH"
    },
    @{
        Name = "LFighter + Non-IID + Label Flipping"
        Aggregation = "lfighter"
        Attack = "label_flipping"
        Partition = "noniid"
        Priority = "HIGH"
    },
    # 实验组2：Non-IID + 动态标签翻转
    @{
        Name = "Mean + Non-IID + Furthest Label Flipping"
        Aggregation = "mean"
        Attack = "furthest_label_flipping"
        Partition = "noniid"
        Priority = "MEDIUM"
    },
    @{
        Name = "Trimmed-Mean + Non-IID + Furthest Label Flipping"
        Aggregation = "trimmed-mean"
        Attack = "furthest_label_flipping"
        Partition = "noniid"
        Priority = "MEDIUM"
    },
    @{
        Name = "FABA + Non-IID + Furthest Label Flipping"
        Aggregation = "faba"
        Attack = "furthest_label_flipping"
        Partition = "noniid"
        Priority = "MEDIUM"
    },
    @{
        Name = "CC + Non-IID + Furthest Label Flipping"
        Aggregation = "cc"
        Attack = "furthest_label_flipping"
        Partition = "noniid"
        Priority = "MEDIUM"
    },
    # 实验组3：IID基线对比
    @{
        Name = "Mean + IID + Label Flipping"
        Aggregation = "mean"
        Attack = "label_flipping"
        Partition = "iid"
        Priority = "LOW"
    },
    @{
        Name = "Trimmed-Mean + IID + Label Flipping"
        Aggregation = "trimmed-mean"
        Attack = "label_flipping"
        Partition = "iid"
        Priority = "LOW"
    },
    @{
        Name = "FABA + IID + Label Flipping"
        Aggregation = "faba"
        Attack = "label_flipping"
        Partition = "iid"
        Priority = "LOW"
    }
)

# 询问用户要运行哪些实验
Write-Host "请选择要运行的实验类型：" -ForegroundColor Yellow
Write-Host "1. 仅运行核心实验（Non-IID + Label Flipping，5个实验）" -ForegroundColor Cyan
Write-Host "2. 运行所有HIGH优先级实验（Non-IID场景，10个实验）" -ForegroundColor Cyan
Write-Host "3. 运行所有实验（13个实验，耗时较长）" -ForegroundColor Cyan
Write-Host ""
$choice = Read-Host "请输入选择 (1/2/3)"

$selectedExperiments = @()
switch ($choice) {
    "1" {
        $selectedExperiments = $experiments | Where-Object { $_.Priority -eq "HIGH" -and $_.Attack -eq "label_flipping" }
        Write-Host "将运行5个核心实验" -ForegroundColor Green
    }
    "2" {
        $selectedExperiments = $experiments | Where-Object { $_.Priority -eq "HIGH" -or ($_.Priority -eq "MEDIUM" -and $_.Aggregation -eq "mean") }
        Write-Host "将运行10个HIGH优先级实验" -ForegroundColor Green
    }
    "3" {
        $selectedExperiments = $experiments
        Write-Host "将运行所有13个实验" -ForegroundColor Green
    }
    default {
        Write-Host "无效选择，默认运行核心实验" -ForegroundColor Yellow
        $selectedExperiments = $experiments | Where-Object { $_.Priority -eq "HIGH" -and $_.Attack -eq "label_flipping" }
    }
}

Write-Host ""
Write-Host "实验列表：" -ForegroundColor Cyan
$experimentNum = 1
foreach ($exp in $selectedExperiments) {
    Write-Host "$experimentNum. $($exp.Name)" -ForegroundColor White
    $experimentNum++
}

Write-Host ""
$confirm = Read-Host "确认开始运行？(Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

# 开始运行实验
$totalExperiments = $selectedExperiments.Count
$currentExperiment = 0
$startTime = Get-Date

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始运行实验" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($exp in $selectedExperiments) {
    $currentExperiment++
    $expStartTime = Get-Date
    
    Write-Host "[$currentExperiment/$totalExperiments] 正在运行: $($exp.Name)" -ForegroundColor Cyan
    Write-Host "  聚合器: $($exp.Aggregation) | 攻击: $($exp.Attack) | 数据分布: $($exp.Partition)" -ForegroundColor Gray
    
    $command = "python `"main CMomentum.py`" --aggregation $($exp.Aggregation) --attack $($exp.Attack) --data-partition $($exp.Partition)"
    
    try {
        # 运行实验
        Invoke-Expression $command
        
        $expEndTime = Get-Date
        $expDuration = $expEndTime - $expStartTime
        Write-Host "  ✓ 完成！耗时: $($expDuration.ToString('mm\:ss'))" -ForegroundColor Green
    }
    catch {
        Write-Host "  ✗ 失败: $_" -ForegroundColor Red
    }
    
    Write-Host ""
    
    # 估算剩余时间
    if ($currentExperiment -lt $totalExperiments) {
        $elapsed = (Get-Date) - $startTime
        $avgTime = $elapsed.TotalSeconds / $currentExperiment
        $remaining = ($totalExperiments - $currentExperiment) * $avgTime
        $remainingTimeSpan = [TimeSpan]::FromSeconds($remaining)
        Write-Host "  预计剩余时间: $($remainingTimeSpan.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
        Write-Host ""
    }
}

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "所有实验完成！" -ForegroundColor Green
Write-Host "总耗时: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "实验结果保存在 ./record 目录" -ForegroundColor Cyan
Write-Host "运行绘图脚本查看结果: cd draw_fig && python draw-MultiFig-Momentum.py" -ForegroundColor Cyan

