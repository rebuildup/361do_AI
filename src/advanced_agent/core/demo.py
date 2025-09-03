"""
Pydantic + Loguru 設定・ログシステム統合デモ
環境検証と設定最適化のデモンストレーション
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.core.config import get_config, load_config, AdvancedAgentConfig
from src.advanced_agent.core.logger import setup_logging, get_logger
from src.advanced_agent.core.environment import validate_environment_startup, quick_environment_check
from src.advanced_agent.core.config_validator import (
    validate_and_optimize_config, generate_system_optimized_config, ConfigValidator
)


async def demo_environment_validation():
    """環境検証デモ"""
    print("\n" + "="*70)
    print("ENVIRONMENT VALIDATION DEMO")
    print("="*70)
    
    # ログシステム初期化
    logger = setup_logging("INFO")
    logger.log_startup("environment_demo", "1.0.0", {"demo": True})
    
    print("\n1. Quick Environment Check")
    print("-" * 30)
    
    quick_result = quick_environment_check()
    print(f"Quick check result: {'✅ PASS' if quick_result else '❌ FAIL'}")
    
    print("\n2. Detailed Environment Validation")
    print("-" * 40)
    
    try:
        # 詳細環境検証（失敗時は例外発生）
        report = validate_environment_startup()
        
        print(f"\n✅ Environment validation completed: {report.overall_status}")
        
        if report.warnings:
            print(f"\n⚠️  Warnings detected ({len(report.warnings)}):")
            for warning in report.warnings[:3]:  # 最初の3つだけ表示
                print(f"   • {warning}")
        
        if report.recommendations:
            print(f"\n💡 Recommendations ({len(report.recommendations)}):")
            for rec in report.recommendations[:3]:  # 最初の3つだけ表示
                print(f"   • {rec}")
        
        # システム情報サマリー
        print(f"\n📊 System Summary:")
        print(f"   Platform: {report.system_info.get('platform', 'Unknown')}")
        print(f"   Python: {report.system_info.get('python_version', 'Unknown')}")
        print(f"   Hostname: {report.system_info.get('hostname', 'Unknown')}")
        
        return True
        
    except RuntimeError as e:
        print(f"\n❌ Environment validation failed: {e}")
        return False


async def demo_config_validation():
    """設定検証デモ"""
    print("\n" + "="*70)
    print("CONFIGURATION VALIDATION DEMO")
    print("="*70)
    
    logger = get_logger()
    
    print("\n1. Current Configuration Validation")
    print("-" * 40)
    
    # 現在の設定ファイル検証
    config_path = "config/advanced_agent.yaml"
    if Path(config_path).exists():
        report = validate_and_optimize_config(config_path, apply_fixes=False)
        
        print(f"Validation Status: {'✅ PASSED' if report.validation_passed else '❌ FAILED'}")
        print(f"Issues Found: {len(report.issues)}")
        print(f"Optimizations Available: {len(report.optimizations)}")
        
        if report.issues:
            print("\nTop Issues:")
            for issue in report.issues[:3]:
                severity_icon = "❌" if issue.severity == "ERROR" else "⚠️"
                print(f"   {severity_icon} {issue.field_path}: {issue.message}")
        
        if report.optimizations:
            print("\nTop Optimizations:")
            for opt in report.optimizations[:3]:
                print(f"   💡 {opt.field_path}: {opt.message}")
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    print("\n2. System-Optimized Configuration Generation")
    print("-" * 50)
    
    # システム最適化設定生成
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            optimized_path = f"{temp_dir}/optimized_config.yaml"
            
            optimized_config = generate_system_optimized_config(optimized_path)
            
            print(f"✅ Generated optimized configuration: {optimized_path}")
            
            # 最適化設定の主要項目表示
            print(f"\n📋 Optimized Settings:")
            print(f"   GPU VRAM Limit: {optimized_config.gpu.max_vram_gb}GB")
            print(f"   CPU Threads: {optimized_config.cpu.max_threads}")
            print(f"   System RAM: {optimized_config.memory.system_ram_gb}GB")
            print(f"   Batch Size: {optimized_config.learning.batch_size}")
            print(f"   Context Length: {optimized_config.models.context_length}")
            
            # 設定ファイル内容の一部表示
            with open(optimized_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            print(f"\n📄 Configuration Preview (first 10 lines):")
            lines = config_content.split('\n')[:10]
            for i, line in enumerate(lines, 1):
                print(f"   {i:2d}: {line}")
            
            total_lines = len(config_content.split('\n'))
            if total_lines > 10:
                print(f"   ... ({total_lines - 10} more lines)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate optimized configuration: {e}")
        return False


async def demo_config_auto_fix():
    """設定自動修正デモ"""
    print("\n" + "="*70)
    print("CONFIGURATION AUTO-FIX DEMO")
    print("="*70)
    
    logger = get_logger()
    
    print("\n1. Creating Test Configuration with Issues")
    print("-" * 50)
    
    # 問題のある設定を作成
    problematic_config = {
        "project_name": "Test Agent",
        "version": "1.0.0",
        "environment": "development",
        "gpu": {
            "max_vram_gb": 12.0,  # 過大なVRAM設定
            "quantization_levels": [4, 3],  # 8bit量子化なし
            "temperature_threshold": 90  # 高い温度閾値
        },
        "cpu": {
            "max_threads": 64,  # 過大なスレッド数
            "offload_threshold": 0.8
        },
        "memory": {
            "system_ram_gb": 128.0,  # 過大なRAM設定
            "cache_size_mb": 4096
        },
        "models": {
            "primary": "deepseek-r1:7b",
            "fallback": "qwen2.5:7b-instruct-q4_k_m",
            "ollama_base_url": "invalid-url",  # 無効なURL
            "context_length": 8192  # 大きなコンテキスト
        },
        "learning": {
            "batch_size": 16,  # 大きなバッチサイズ
            "learning_rate": 0.01  # 高い学習率
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(problematic_config, f, default_flow_style=False)
        test_config_path = f.name
    
    try:
        print(f"Created test configuration: {test_config_path}")
        
        print("\n2. Initial Validation (Before Auto-Fix)")
        print("-" * 45)
        
        # 初期検証
        initial_report = validate_and_optimize_config(test_config_path, apply_fixes=False)
        
        print(f"Validation Status: {'✅ PASSED' if initial_report.validation_passed else '❌ FAILED'}")
        print(f"Issues Found: {len(initial_report.issues)}")
        print(f"Auto-fixable Issues: {sum(1 for issue in initial_report.issues + initial_report.optimizations if issue.auto_fixable)}")
        
        if initial_report.issues:
            print("\nIssues Found:")
            for issue in initial_report.issues:
                fixable = "🔧" if issue.auto_fixable else "⚠️"
                print(f"   {fixable} {issue.field_path}: {issue.message}")
        
        print("\n3. Applying Auto-Fixes")
        print("-" * 25)
        
        # 自動修正適用
        validator = ConfigValidator()
        success, fixes = validator.apply_auto_fixes(test_config_path, initial_report)
        
        if success and fixes:
            print(f"✅ Applied {len(fixes)} auto-fixes:")
            for fix in fixes:
                print(f"   🔧 {fix}")
        else:
            print("❌ No auto-fixes applied or failed to apply")
        
        print("\n4. Post-Fix Validation")
        print("-" * 25)
        
        # 修正後検証
        final_report = validate_and_optimize_config(test_config_path, apply_fixes=False)
        
        print(f"Validation Status: {'✅ PASSED' if final_report.validation_passed else '❌ FAILED'}")
        print(f"Remaining Issues: {len(final_report.issues)}")
        print(f"Remaining Optimizations: {len(final_report.optimizations)}")
        
        # 修正前後の比較
        print(f"\n📊 Improvement Summary:")
        print(f"   Issues: {len(initial_report.issues)} → {len(final_report.issues)}")
        print(f"   Auto-fixes Applied: {len(fixes) if fixes else 0}")
        
        # 修正後の設定内容確認
        with open(test_config_path, 'r', encoding='utf-8') as f:
            fixed_config = yaml.safe_load(f)
        
        print(f"\n📋 Key Settings After Auto-Fix:")
        print(f"   GPU VRAM: {fixed_config['gpu']['max_vram_gb']}GB")
        print(f"   CPU Threads: {fixed_config['cpu']['max_threads']}")
        print(f"   System RAM: {fixed_config['memory']['system_ram_gb']}GB")
        print(f"   Batch Size: {fixed_config['learning']['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-fix demo failed: {e}")
        return False
        
    finally:
        # テストファイル削除
        Path(test_config_path).unlink(missing_ok=True)
        # バックアップファイルも削除
        for backup_file in Path(test_config_path).parent.glob(f"{Path(test_config_path).name}.backup_*"):
            backup_file.unlink()


async def demo_logging_integration():
    """ログ統合デモ"""
    print("\n" + "="*70)
    print("LOGGING INTEGRATION DEMO")
    print("="*70)
    
    print("\n1. Structured Logging Examples")
    print("-" * 35)
    
    logger = get_logger()
    
    # 各種ログの例
    logger.log_startup("demo_component", "1.0.0", {
        "environment": "demo",
        "features": ["config_validation", "environment_check"]
    })
    
    logger.log_system_stats({
        "cpu_percent": 45.2,
        "memory_percent": 67.8,
        "disk_usage": 23.1
    })
    
    logger.log_gpu_stats({
        "gpu_memory_percent": 34.5,
        "gpu_utilization": 78.2,
        "gpu_temperature": 65
    })
    
    logger.log_config_change(
        config_section="gpu.max_vram_gb",
        old_value=6.0,
        new_value=5.0,
        changed_by="auto_optimizer"
    )
    
    logger.log_alert(
        alert_type="config_optimization",
        severity="INFO",
        message="Configuration optimized for RTX 4050",
        metadata={"optimizations_applied": 3}
    )
    
    logger.log_performance_metric(
        metric_name="config_validation_time",
        value=0.125,
        unit="seconds",
        component="config_validator"
    )
    
    print("✅ Generated various structured log entries")
    
    print("\n2. Log File Locations")
    print("-" * 25)
    
    config = get_config()
    logs_dir = config.get_logs_dir()
    
    print(f"📁 Logs Directory: {logs_dir}")
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.json"))
        if log_files:
            print(f"📄 Available Log Files ({len(log_files)}):")
            for log_file in sorted(log_files)[-5:]:  # 最新5ファイル
                size_kb = log_file.stat().st_size / 1024
                print(f"   • {log_file.name} ({size_kb:.1f} KB)")
        else:
            print("📄 No log files found yet")
    else:
        print("📁 Logs directory not created yet")
    
    return True


async def main():
    """メインデモ実行"""
    print("ADVANCED AGENT - PYDANTIC + LOGURU INTEGRATION DEMO")
    print("=" * 70)
    print("RTX 4050 6GB VRAM 最適化システム")
    print("設定管理・環境検証・ログシステム統合デモンストレーション")
    
    results = []
    
    try:
        # 1. 環境検証デモ
        env_result = await demo_environment_validation()
        results.append(("Environment Validation", env_result))
        
        # 2. 設定検証デモ
        config_result = await demo_config_validation()
        results.append(("Configuration Validation", config_result))
        
        # 3. 自動修正デモ
        autofix_result = await demo_config_auto_fix()
        results.append(("Configuration Auto-Fix", autofix_result))
        
        # 4. ログ統合デモ
        logging_result = await demo_logging_integration()
        results.append(("Logging Integration", logging_result))
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 結果サマリー
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n📊 Overall Results: {success_count}/{total_count} demos successful")
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"   {status} {demo_name}")
    
    if success_count == total_count:
        print(f"\n🎉 All demos completed successfully!")
        print(f"   • Environment validation system working")
        print(f"   • Configuration validation and optimization working")
        print(f"   • Auto-fix functionality working")
        print(f"   • Structured logging system working")
    else:
        print(f"\n⚠️  {total_count - success_count} demo(s) failed")
        print(f"   Please check the error messages above")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Check generated log files in logs/ directory")
    print(f"   • Review optimized configuration suggestions")
    print(f"   • Run environment validation before starting the agent")
    print(f"   • Use auto-fix to optimize your configuration")
    
    # 最終ログ
    logger = get_logger()
    logger.log_shutdown(
        component="pydantic_loguru_demo",
        uptime_seconds=0,  # デモなので0
        final_stats={
            "demos_run": total_count,
            "demos_successful": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0
        }
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()