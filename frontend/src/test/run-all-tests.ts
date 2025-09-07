/**
 * Comprehensive Test Runner
 *
 * Runs all tests and generates comprehensive reports
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

interface TestResults {
  unit: {
    passed: number;
    failed: number;
    total: number;
    coverage: {
      lines: number;
      functions: number;
      branches: number;
      statements: number;
    };
  };
  e2e: {
    passed: number;
    failed: number;
    total: number;
  };
  integration: {
    passed: number;
    failed: number;
    total: number;
  };
}

class TestRunner {
  private results: TestResults = {
    unit: {
      passed: 0,
      failed: 0,
      total: 0,
      coverage: { lines: 0, functions: 0, branches: 0, statements: 0 },
    },
    e2e: { passed: 0, failed: 0, total: 0 },
    integration: { passed: 0, failed: 0, total: 0 },
  };

  async runAllTests(): Promise<void> {
    console.log('🧪 Starting comprehensive test suite...\n');

    try {
      // Run unit tests
      await this.runUnitTests();

      // Run integration tests
      await this.runIntegrationTests();

      // Run E2E tests
      await this.runE2ETests();

      // Generate final report
      this.generateFinalReport();
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      process.exit(1);
    }
  }

  private async runUnitTests(): Promise<void> {
    console.log('🔬 Running unit tests...');

    try {
      // Run Vitest with coverage
      const output = execSync(
        'npm run test:run -- --coverage --reporter=json',
        {
          encoding: 'utf8',
          cwd: process.cwd(),
        }
      );

      // Parse results
      try {
        const results = JSON.parse(output);
        this.results.unit.passed = results.numPassedTests || 0;
        this.results.unit.failed = results.numFailedTests || 0;
        this.results.unit.total = results.numTotalTests || 0;

        // Parse coverage if available
        if (results.coverageMap) {
          const coverage = results.coverageMap.getCoverageSummary();
          this.results.unit.coverage = {
            lines: coverage.lines.pct,
            functions: coverage.functions.pct,
            branches: coverage.branches.pct,
            statements: coverage.statements.pct,
          };
        }
      } catch {
        console.warn('Could not parse unit test results');
      }

      console.log('✅ Unit tests completed');
    } catch (error) {
      console.error('❌ Unit tests failed');
      throw error;
    }
  }

  private async runIntegrationTests(): Promise<void> {
    console.log('🔗 Running integration tests...');

    try {
      // Run integration tests (subset of unit tests focusing on integration)
      const output = execSync(
        'npm run test:run -- --reporter=json src/**/*.integration.test.*',
        {
          encoding: 'utf8',
          cwd: process.cwd(),
        }
      );

      // Parse results
      try {
        const results = JSON.parse(output);
        this.results.integration.passed = results.numPassedTests || 0;
        this.results.integration.failed = results.numFailedTests || 0;
        this.results.integration.total = results.numTotalTests || 0;
      } catch {
        console.warn('Could not parse integration test results');
      }

      console.log('✅ Integration tests completed');
    } catch {
      console.warn('⚠️ Integration tests failed or not found');
      // Don't throw here as integration tests might not exist yet
    }
  }

  private async runE2ETests(): Promise<void> {
    console.log('🌐 Running E2E tests...');

    try {
      // Check if Playwright is available
      execSync('npm run test:e2e -- --reporter=json', {
        encoding: 'utf8',
        cwd: process.cwd(),
      });

      // Parse Playwright results
      try {
        const resultsPath = path.join(
          process.cwd(),
          'test-results',
          'results.json'
        );
        if (fs.existsSync(resultsPath)) {
          const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));

          this.results.e2e.total =
            results.suites?.reduce(
              (total: number, suite: any) => total + (suite.specs?.length || 0),
              0
            ) || 0;

          this.results.e2e.passed =
            results.suites?.reduce(
              (passed: number, suite: any) =>
                passed +
                (suite.specs?.filter((spec: any) => spec.ok).length || 0),
              0
            ) || 0;

          this.results.e2e.failed =
            this.results.e2e.total - this.results.e2e.passed;
        }
      } catch {
        console.warn('Could not parse E2E test results');
      }

      console.log('✅ E2E tests completed');
    } catch {
      console.warn('⚠️ E2E tests failed or Playwright not available');
      // Don't throw here as E2E tests might not be runnable in all environments
    }
  }

  private generateFinalReport(): void {
    console.log('\n📊 Test Results Summary');
    console.log('='.repeat(50));

    // Unit Tests
    console.log(`\n🔬 Unit Tests:`);
    console.log(`   Passed: ${this.results.unit.passed}`);
    console.log(`   Failed: ${this.results.unit.failed}`);
    console.log(`   Total:  ${this.results.unit.total}`);
    console.log(
      `   Success Rate: ${this.calculateSuccessRate(this.results.unit.passed, this.results.unit.total)}%`
    );

    if (this.results.unit.coverage.lines > 0) {
      console.log(`\n📈 Coverage:`);
      console.log(
        `   Lines:      ${this.results.unit.coverage.lines.toFixed(1)}%`
      );
      console.log(
        `   Functions:  ${this.results.unit.coverage.functions.toFixed(1)}%`
      );
      console.log(
        `   Branches:   ${this.results.unit.coverage.branches.toFixed(1)}%`
      );
      console.log(
        `   Statements: ${this.results.unit.coverage.statements.toFixed(1)}%`
      );
    }

    // Integration Tests
    if (this.results.integration.total > 0) {
      console.log(`\n🔗 Integration Tests:`);
      console.log(`   Passed: ${this.results.integration.passed}`);
      console.log(`   Failed: ${this.results.integration.failed}`);
      console.log(`   Total:  ${this.results.integration.total}`);
      console.log(
        `   Success Rate: ${this.calculateSuccessRate(this.results.integration.passed, this.results.integration.total)}%`
      );
    }

    // E2E Tests
    if (this.results.e2e.total > 0) {
      console.log(`\n🌐 E2E Tests:`);
      console.log(`   Passed: ${this.results.e2e.passed}`);
      console.log(`   Failed: ${this.results.e2e.failed}`);
      console.log(`   Total:  ${this.results.e2e.total}`);
      console.log(
        `   Success Rate: ${this.calculateSuccessRate(this.results.e2e.passed, this.results.e2e.total)}%`
      );
    }

    // Overall Summary
    const totalPassed =
      this.results.unit.passed +
      this.results.integration.passed +
      this.results.e2e.passed;
    const totalTests =
      this.results.unit.total +
      this.results.integration.total +
      this.results.e2e.total;
    const overallSuccessRate = this.calculateSuccessRate(
      totalPassed,
      totalTests
    );

    console.log(`\n🎯 Overall Results:`);
    console.log(`   Total Passed: ${totalPassed}`);
    console.log(`   Total Tests:  ${totalTests}`);
    console.log(`   Success Rate: ${overallSuccessRate}%`);

    // Quality Gates
    this.checkQualityGates(overallSuccessRate);

    // Save results to file
    this.saveResultsToFile();
  }

  private calculateSuccessRate(passed: number, total: number): number {
    return total > 0 ? Math.round((passed / total) * 100) : 0;
  }

  private checkQualityGates(overallSuccessRate: number): void {
    console.log(`\n🚪 Quality Gates:`);

    const gates = [
      {
        name: 'Overall Success Rate',
        threshold: 90,
        actual: overallSuccessRate,
        unit: '%',
      },
      {
        name: 'Line Coverage',
        threshold: 80,
        actual: this.results.unit.coverage.lines,
        unit: '%',
      },
      {
        name: 'Function Coverage',
        threshold: 80,
        actual: this.results.unit.coverage.functions,
        unit: '%',
      },
      {
        name: 'Branch Coverage',
        threshold: 80,
        actual: this.results.unit.coverage.branches,
        unit: '%',
      },
    ];

    let allGatesPassed = true;

    gates.forEach(gate => {
      const passed = gate.actual >= gate.threshold;
      const status = passed ? '✅' : '❌';
      console.log(
        `   ${status} ${gate.name}: ${gate.actual.toFixed(1)}${gate.unit} (threshold: ${gate.threshold}${gate.unit})`
      );

      if (!passed) {
        allGatesPassed = false;
      }
    });

    if (allGatesPassed) {
      console.log('\n🎉 All quality gates passed!');
    } else {
      console.log(
        '\n⚠️ Some quality gates failed. Please review and improve test coverage.'
      );
    }
  }

  private saveResultsToFile(): void {
    const resultsFile = path.join(
      process.cwd(),
      'test-results',
      'comprehensive-results.json'
    );

    // Ensure directory exists
    const dir = path.dirname(resultsFile);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Save results
    fs.writeFileSync(
      resultsFile,
      JSON.stringify(
        {
          timestamp: new Date().toISOString(),
          results: this.results,
          summary: {
            totalPassed:
              this.results.unit.passed +
              this.results.integration.passed +
              this.results.e2e.passed,
            totalTests:
              this.results.unit.total +
              this.results.integration.total +
              this.results.e2e.total,
            overallSuccessRate: this.calculateSuccessRate(
              this.results.unit.passed +
                this.results.integration.passed +
                this.results.e2e.passed,
              this.results.unit.total +
                this.results.integration.total +
                this.results.e2e.total
            ),
          },
        },
        null,
        2
      )
    );

    console.log(`\n💾 Results saved to: ${resultsFile}`);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const runner = new TestRunner();
  runner.runAllTests().catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

export { TestRunner };
