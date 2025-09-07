#!/usr/bin/env node
/**
 * Production Build Test Script
 * プロダクションビルドテストスクリプト
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Color codes for output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
};

// Logging functions
const log = message => {
  console.log(
    `${colors.green}[${new Date().toISOString()}] ${message}${colors.reset}`
  );
};

const warn = message => {
  console.log(
    `${colors.yellow}[${new Date().toISOString()}] WARNING: ${message}${colors.reset}`
  );
};

const error = message => {
  console.log(
    `${colors.red}[${new Date().toISOString()}] ERROR: ${message}${colors.reset}`
  );
};

// Test configuration
const DIST_DIR = path.join(__dirname, '..', 'dist');
const REQUIRED_FILES = ['index.html', 'assets'];

const REQUIRED_PATTERNS = [
  /assets\/index-[a-f0-9]+\.js$/,
  /assets\/index-[a-f0-9]+\.css$/,
];

// Main test function
async function testProductionBuild() {
  try {
    log('Starting production build test...');

    // Check if we're in the frontend directory
    const packageJsonPath = path.join(__dirname, '..', 'package.json');
    if (!fs.existsSync(packageJsonPath)) {
      throw new Error(
        "package.json not found. Make sure you're in the frontend directory."
      );
    }

    // Clean previous build
    if (fs.existsSync(DIST_DIR)) {
      log('Cleaning previous build...');
      fs.rmSync(DIST_DIR, { recursive: true, force: true });
    }

    // Run build
    log('Running production build...');
    execSync('npm run build', {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..'),
    });

    // Check if dist directory exists
    if (!fs.existsSync(DIST_DIR)) {
      throw new Error('Build output directory (dist) not found');
    }

    log('Build completed. Verifying output...');

    // Check required files
    for (const file of REQUIRED_FILES) {
      const filePath = path.join(DIST_DIR, file);
      if (!fs.existsSync(filePath)) {
        throw new Error(`Required file/directory missing: ${file}`);
      }
      log(`✓ Found required file: ${file}`);
    }

    // Check for hashed assets
    const assetsDir = path.join(DIST_DIR, 'assets');
    if (fs.existsSync(assetsDir)) {
      const assetFiles = fs.readdirSync(assetsDir);

      for (const pattern of REQUIRED_PATTERNS) {
        const found = assetFiles.some(file => pattern.test(`assets/${file}`));
        if (!found) {
          warn(`Expected asset pattern not found: ${pattern}`);
        } else {
          log(`✓ Found expected asset pattern: ${pattern}`);
        }
      }
    }

    // Check index.html content
    const indexPath = path.join(DIST_DIR, 'index.html');
    const indexContent = fs.readFileSync(indexPath, 'utf8');

    // Verify essential HTML structure
    const requiredElements = [
      '<div id="root">',
      '<script type="module"',
      '<link rel="stylesheet"',
    ];

    for (const element of requiredElements) {
      if (!indexContent.includes(element)) {
        throw new Error(`Required HTML element missing: ${element}`);
      }
      log(`✓ Found required HTML element: ${element}`);
    }

    // Calculate build size
    const getBuildSize = dir => {
      let size = 0;
      const files = fs.readdirSync(dir);

      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
          size += getBuildSize(filePath);
        } else {
          size += stat.size;
        }
      }

      return size;
    };

    const buildSize = getBuildSize(DIST_DIR);
    const buildSizeMB = (buildSize / 1024 / 1024).toFixed(2);

    log(`Build size: ${buildSizeMB} MB`);

    // Check if build size is reasonable (warn if > 10MB)
    if (buildSize > 10 * 1024 * 1024) {
      warn(`Build size is quite large: ${buildSizeMB} MB`);
    }

    // List all files in dist
    log('Build output structure:');
    const listFiles = (dir, prefix = '') => {
      const files = fs.readdirSync(dir);

      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
          console.log(`${prefix}📁 ${file}/`);
          listFiles(filePath, prefix + '  ');
        } else {
          const sizeMB = (stat.size / 1024).toFixed(1);
          console.log(`${prefix}📄 ${file} (${sizeMB} KB)`);
        }
      }
    };

    listFiles(DIST_DIR);

    log('✅ Production build test completed successfully!');
    log(`Build output is ready in: ${DIST_DIR}`);

    return true;
  } catch (err) {
    error(`Production build test failed: ${err.message}`);
    return false;
  }
}

// Run the test
if (require.main === module) {
  testProductionBuild()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(err => {
      error(`Unexpected error: ${err.message}`);
      process.exit(1);
    });
}

module.exports = { testProductionBuild };
