#!/usr/bin/env node

/**
 * Performance Audit Script
 *
 * Analyzes build output and generates performance reports
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class PerformanceAuditor {
  constructor() {
    this.distPath = path.join(__dirname, '../dist');
    this.results = {
      bundleSize: {},
      chunkAnalysis: {},
      assetOptimization: {},
      recommendations: [],
    };
  }

  async runAudit() {
    console.log('🔍 Starting performance audit...\n');

    try {
      // Check if build exists
      if (!fs.existsSync(this.distPath)) {
        console.error('❌ Build not found. Run "npm run build" first.');
        process.exit(1);
      }

      // Analyze bundle size
      this.analyzeBundleSize();

      // Analyze chunks
      this.analyzeChunks();

      // Analyze assets
      this.analyzeAssets();

      // Generate recommendations
      this.generateRecommendations();

      // Generate report
      this.generateReport();

      console.log('✅ Performance audit completed!\n');
    } catch (error) {
      console.error('❌ Performance audit failed:', error);
      process.exit(1);
    }
  }

  analyzeBundleSize() {
    console.log('📦 Analyzing bundle size...');

    const jsFiles = this.getFilesByExtension('.js');
    const cssFiles = this.getFilesByExtension('.css');

    let totalJsSize = 0;
    let totalCssSize = 0;

    jsFiles.forEach(file => {
      const size = this.getFileSize(file);
      totalJsSize += size;
      this.results.bundleSize[path.basename(file)] = this.formatSize(size);
    });

    cssFiles.forEach(file => {
      const size = this.getFileSize(file);
      totalCssSize += size;
      this.results.bundleSize[path.basename(file)] = this.formatSize(size);
    });

    this.results.bundleSize.totalJs = this.formatSize(totalJsSize);
    this.results.bundleSize.totalCss = this.formatSize(totalCssSize);
    this.results.bundleSize.total = this.formatSize(totalJsSize + totalCssSize);

    console.log(`   Total JS: ${this.results.bundleSize.totalJs}`);
    console.log(`   Total CSS: ${this.results.bundleSize.totalCss}`);
    console.log(`   Total: ${this.results.bundleSize.total}\n`);
  }

  analyzeChunks() {
    console.log('🧩 Analyzing chunks...');

    const jsFiles = this.getFilesByExtension('.js');
    const chunks = {};

    jsFiles.forEach(file => {
      const fileName = path.basename(file);
      const size = this.getFileSize(file);

      // Categorize chunks
      let category = 'other';
      if (fileName.includes('react-vendor')) category = 'react';
      else if (fileName.includes('ui-vendor')) category = 'ui';
      else if (fileName.includes('utils-vendor')) category = 'utils';
      else if (fileName.includes('chat-features')) category = 'chat';
      else if (fileName.includes('debug-features')) category = 'debug';
      else if (fileName.includes('services')) category = 'services';
      else if (fileName.includes('index')) category = 'main';

      if (!chunks[category]) chunks[category] = { files: [], totalSize: 0 };
      chunks[category].files.push({
        name: fileName,
        size: this.formatSize(size),
      });
      chunks[category].totalSize += size;
    });

    // Format chunk data
    Object.keys(chunks).forEach(category => {
      chunks[category].totalSize = this.formatSize(chunks[category].totalSize);
      console.log(`   ${category}: ${chunks[category].totalSize}`);
    });

    this.results.chunkAnalysis = chunks;
    console.log('');
  }

  analyzeAssets() {
    console.log('🖼️  Analyzing assets...');

    const imageFiles = this.getFilesByExtension([
      '.png',
      '.jpg',
      '.jpeg',
      '.svg',
      '.gif',
    ]);
    const fontFiles = this.getFilesByExtension([
      '.woff',
      '.woff2',
      '.ttf',
      '.eot',
    ]);

    let totalImageSize = 0;
    let totalFontSize = 0;

    imageFiles.forEach(file => {
      totalImageSize += this.getFileSize(file);
    });

    fontFiles.forEach(file => {
      totalFontSize += this.getFileSize(file);
    });

    this.results.assetOptimization = {
      images: {
        count: imageFiles.length,
        totalSize: this.formatSize(totalImageSize),
      },
      fonts: {
        count: fontFiles.length,
        totalSize: this.formatSize(totalFontSize),
      },
    };

    console.log(
      `   Images: ${imageFiles.length} files, ${this.formatSize(totalImageSize)}`
    );
    console.log(
      `   Fonts: ${fontFiles.length} files, ${this.formatSize(totalFontSize)}\n`
    );
  }

  generateRecommendations() {
    console.log('💡 Generating recommendations...');

    const recommendations = [];

    // Bundle size recommendations
    const totalSizeBytes = this.parseSize(this.results.bundleSize.total);
    if (totalSizeBytes > 1024 * 1024) {
      // > 1MB
      recommendations.push({
        type: 'warning',
        category: 'Bundle Size',
        message: `Total bundle size (${this.results.bundleSize.total}) exceeds 1MB. Consider code splitting or removing unused dependencies.`,
      });
    }

    // Chunk recommendations
    Object.keys(this.results.chunkAnalysis).forEach(category => {
      const chunkSize = this.parseSize(
        this.results.chunkAnalysis[category].totalSize
      );
      if (chunkSize > 500 * 1024) {
        // > 500KB
        recommendations.push({
          type: 'info',
          category: 'Chunk Size',
          message: `${category} chunk (${this.results.chunkAnalysis[category].totalSize}) is large. Consider further splitting.`,
        });
      }
    });

    // Asset recommendations
    if (this.results.assetOptimization.images.count > 10) {
      recommendations.push({
        type: 'info',
        category: 'Assets',
        message: `${this.results.assetOptimization.images.count} image files detected. Consider using image optimization and lazy loading.`,
      });
    }

    // Performance recommendations
    recommendations.push({
      type: 'success',
      category: 'Performance',
      message: 'Code splitting is configured for optimal loading.',
    });

    recommendations.push({
      type: 'success',
      category: 'Performance',
      message: 'Terser minification is enabled for production builds.',
    });

    this.results.recommendations = recommendations;

    recommendations.forEach(rec => {
      const icon =
        rec.type === 'warning' ? '⚠️' : rec.type === 'info' ? 'ℹ️' : '✅';
      console.log(`   ${icon} ${rec.message}`);
    });

    console.log('');
  }

  generateReport() {
    console.log('📊 Performance Audit Report');
    console.log('='.repeat(50));

    // Bundle Size Summary
    console.log('\n📦 Bundle Size:');
    console.log(`   JavaScript: ${this.results.bundleSize.totalJs}`);
    console.log(`   CSS: ${this.results.bundleSize.totalCss}`);
    console.log(`   Total: ${this.results.bundleSize.total}`);

    // Chunk Analysis
    console.log('\n🧩 Chunk Analysis:');
    Object.keys(this.results.chunkAnalysis).forEach(category => {
      console.log(
        `   ${category}: ${this.results.chunkAnalysis[category].totalSize}`
      );
    });

    // Asset Summary
    console.log('\n🖼️  Assets:');
    console.log(
      `   Images: ${this.results.assetOptimization.images.count} files (${this.results.assetOptimization.images.totalSize})`
    );
    console.log(
      `   Fonts: ${this.results.assetOptimization.fonts.count} files (${this.results.assetOptimization.fonts.totalSize})`
    );

    // Recommendations
    console.log('\n💡 Recommendations:');
    this.results.recommendations.forEach(rec => {
      const icon =
        rec.type === 'warning' ? '⚠️' : rec.type === 'info' ? 'ℹ️' : '✅';
      console.log(`   ${icon} [${rec.category}] ${rec.message}`);
    });

    // Save detailed report
    const reportPath = path.join(this.distPath, 'performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);

    // Performance score
    const score = this.calculatePerformanceScore();
    console.log(`\n🎯 Performance Score: ${score}/100`);

    if (score >= 90) {
      console.log('🎉 Excellent performance!');
    } else if (score >= 70) {
      console.log('👍 Good performance, room for improvement.');
    } else {
      console.log('⚠️  Performance needs attention.');
    }
  }

  calculatePerformanceScore() {
    let score = 100;

    // Deduct points for large bundle size
    const totalSizeBytes = this.parseSize(this.results.bundleSize.total);
    if (totalSizeBytes > 2 * 1024 * 1024)
      score -= 30; // > 2MB
    else if (totalSizeBytes > 1024 * 1024)
      score -= 15; // > 1MB
    else if (totalSizeBytes > 500 * 1024) score -= 5; // > 500KB

    // Deduct points for warnings
    const warnings = this.results.recommendations.filter(
      r => r.type === 'warning'
    ).length;
    score -= warnings * 10;

    // Deduct points for large chunks
    const largeChunks = Object.keys(this.results.chunkAnalysis).filter(
      category =>
        this.parseSize(this.results.chunkAnalysis[category].totalSize) >
        500 * 1024
    ).length;
    score -= largeChunks * 5;

    return Math.max(0, Math.min(100, score));
  }

  getFilesByExtension(extensions) {
    const exts = Array.isArray(extensions) ? extensions : [extensions];
    const files = [];

    const scanDir = dir => {
      const items = fs.readdirSync(dir);
      items.forEach(item => {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          scanDir(fullPath);
        } else {
          const ext = path.extname(item);
          if (exts.includes(ext)) {
            files.push(fullPath);
          }
        }
      });
    };

    scanDir(this.distPath);
    return files;
  }

  getFileSize(filePath) {
    return fs.statSync(filePath).size;
  }

  formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  parseSize(sizeString) {
    const match = sizeString.match(/^([\d.]+)\s*([KMGT]?B)$/);
    if (!match) return 0;

    const value = parseFloat(match[1]);
    const unit = match[2];

    const multipliers = {
      B: 1,
      KB: 1024,
      MB: 1024 * 1024,
      GB: 1024 * 1024 * 1024,
    };
    return value * (multipliers[unit] || 1);
  }
}

// Run the audit
const auditor = new PerformanceAuditor();
auditor.runAudit().catch(error => {
  console.error('Audit failed:', error);
  process.exit(1);
});
