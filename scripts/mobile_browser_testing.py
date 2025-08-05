#!/usr/bin/env python3
"""
Mobile Responsiveness and Cross-Browser Testing Script
Tests frontend responsiveness, mobile features, and browser compatibility
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright
import time

class MobileResponsivenessTester:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.results = {
            'mobile_tests': [],
            'browser_tests': [],
            'accessibility_tests': [],
            'performance_tests': []
        }
        
        # Device configurations for testing
        self.mobile_devices = [
            {'name': 'iPhone 13', 'width': 390, 'height': 844, 'deviceScaleFactor': 3, 'isMobile': True},
            {'name': 'iPhone 13 Pro Max', 'width': 428, 'height': 926, 'deviceScaleFactor': 3, 'isMobile': True},
            {'name': 'Samsung Galaxy S21', 'width': 384, 'height': 854, 'deviceScaleFactor': 2.75, 'isMobile': True},
            {'name': 'iPad Air', 'width': 820, 'height': 1180, 'deviceScaleFactor': 2, 'isMobile': True},
            {'name': 'iPad Pro', 'width': 1024, 'height': 1366, 'deviceScaleFactor': 2, 'isMobile': True},
        ]
        
        # Browser configurations
        self.browsers = ['chromium', 'firefox', 'webkit']  # webkit = Safari
        
        # Key pages to test
        self.test_pages = [
            {'path': '/', 'name': 'Home Page'},
            {'path': '/chat', 'name': 'Chat Interface'},
            {'path': '/creative', 'name': 'Creative Evolution'},
            {'path': '/memories', 'name': 'Memory Browser'},
            {'path': '/settings', 'name': 'Settings Page'},
        ]

    async def test_mobile_responsiveness(self):
        """Test mobile responsiveness across different devices"""
        print("📱 Testing Mobile Responsiveness...")
        
        async with async_playwright() as p:
            for browser_name in ['chromium']:  # Start with Chrome for mobile testing
                browser = await p.chromium.launch(headless=True)
                
                for device in self.mobile_devices:
                    print(f"  Testing on {device['name']}...")
                    
                    context = await browser.new_context(
                        viewport={'width': device['width'], 'height': device['height']},
                        device_scale_factor=device['deviceScaleFactor'],
                        is_mobile=device['isMobile']
                    )
                    
                    page = await context.new_page()
                    
                    for test_page in self.test_pages:
                        start_time = time.time()
                        try:
                            # Navigate to page
                            await page.goto(f"{self.base_url}{test_page['path']}")
                            await page.wait_for_load_state('networkidle', timeout=10000)
                            
                            load_time = (time.time() - start_time) * 1000
                            
                            # Test mobile-specific elements
                            mobile_test_results = await self.run_mobile_tests(page, device)
                            
                            self.results['mobile_tests'].append({
                                'device': device['name'],
                                'page': test_page['name'],
                                'load_time': load_time,
                                'viewport': f"{device['width']}x{device['height']}",
                                'tests': mobile_test_results,
                                'success': all(test.get('passed', False) for test in mobile_test_results),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            status = "✅" if all(test.get('passed', False) for test in mobile_test_results) else "⚠️"
                            print(f"    {status} {test_page['name']}: {load_time:.0f}ms")
                            
                        except Exception as e:
                            print(f"    ❌ {test_page['name']}: {str(e)}")
                            self.results['mobile_tests'].append({
                                'device': device['name'],
                                'page': test_page['name'],
                                'error': str(e),
                                'success': False,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    await context.close()
                
                await browser.close()

    async def run_mobile_tests(self, page, device):
        """Run specific mobile responsiveness tests on a page"""
        tests = []
        
        try:
            # Test 1: Check if navigation is mobile-friendly
            nav_elements = await page.query_selector_all('nav, .navigation, .menu')
            mobile_nav_exists = len(nav_elements) > 0
            
            # Check for hamburger menu on smaller screens
            if device['width'] < 768:
                hamburger = await page.query_selector('.hamburger, .menu-toggle, [aria-label*="menu"]')
                mobile_nav_accessible = hamburger is not None
            else:
                mobile_nav_accessible = True
            
            tests.append({
                'name': 'Mobile Navigation',
                'passed': mobile_nav_exists and mobile_nav_accessible,
                'details': f"Nav elements: {len(nav_elements)}, Mobile accessible: {mobile_nav_accessible}"
            })
            
            # Test 2: Check touch targets are large enough (minimum 44px)
            buttons = await page.query_selector_all('button, .btn, [role="button"]')
            touch_target_sizes = []
            
            for button in buttons[:10]:  # Test first 10 buttons
                box = await button.bounding_box()
                if box:
                    min_dimension = min(box['width'], box['height'])
                    touch_target_sizes.append(min_dimension >= 44)
            
            adequate_touch_targets = all(touch_target_sizes) if touch_target_sizes else True
            
            tests.append({
                'name': 'Touch Target Size',
                'passed': adequate_touch_targets,
                'details': f"Buttons tested: {len(touch_target_sizes)}, Adequate: {sum(touch_target_sizes)}"
            })
            
            # Test 3: Check text readability (no text smaller than 16px on mobile)
            if device['isMobile']:
                small_text = await page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('*');
                        let smallTextCount = 0;
                        let totalTextElements = 0;
                        
                        elements.forEach(el => {
                            if (el.textContent && el.textContent.trim()) {
                                const fontSize = window.getComputedStyle(el).fontSize;
                                const sizeInPx = parseFloat(fontSize);
                                
                                if (sizeInPx > 0) {
                                    totalTextElements++;
                                    if (sizeInPx < 16) {
                                        smallTextCount++;
                                    }
                                }
                            }
                        });
                        
                        return { smallTextCount, totalTextElements };
                    }
                """)
                
                readable_text = small_text['smallTextCount'] / max(small_text['totalTextElements'], 1) < 0.1
                
                tests.append({
                    'name': 'Text Readability',
                    'passed': readable_text,
                    'details': f"Small text: {small_text['smallTextCount']}/{small_text['totalTextElements']}"
                })
            
            # Test 4: Check horizontal scrolling (should not exist)
            no_horizontal_scroll = await page.evaluate("""
                () => document.documentElement.scrollWidth <= window.innerWidth
            """)
            
            tests.append({
                'name': 'No Horizontal Scroll',
                'passed': no_horizontal_scroll,
                'details': f"Page width fits viewport: {no_horizontal_scroll}"
            })
            
            # Test 5: Check form elements are mobile-friendly
            inputs = await page.query_selector_all('input, textarea, select')
            mobile_friendly_forms = True
            
            for input_elem in inputs[:5]:  # Test first 5 inputs
                box = await input_elem.bounding_box()
                if box and box['height'] < 40:  # Minimum touch-friendly height
                    mobile_friendly_forms = False
                    break
            
            tests.append({
                'name': 'Mobile-Friendly Forms',
                'passed': mobile_friendly_forms,
                'details': f"Form elements tested: {min(len(inputs), 5)}"
            })
            
        except Exception as e:
            tests.append({
                'name': 'Mobile Test Suite',
                'passed': False,
                'error': str(e)
            })
        
        return tests

    async def test_cross_browser_compatibility(self):
        """Test cross-browser compatibility"""
        print("🌐 Testing Cross-Browser Compatibility...")
        
        async with async_playwright() as p:
            for browser_name in self.browsers:
                print(f"  Testing {browser_name.title()}...")
                
                if browser_name == 'chromium':
                    browser = await p.chromium.launch(headless=True)
                elif browser_name == 'firefox':
                    browser = await p.firefox.launch(headless=True)
                elif browser_name == 'webkit':
                    browser = await p.webkit.launch(headless=True)
                
                context = await browser.new_context()
                page = await context.new_page()
                
                for test_page in self.test_pages:
                    start_time = time.time()
                    try:
                        await page.goto(f"{self.base_url}{test_page['path']}")
                        await page.wait_for_load_state('networkidle', timeout=10000)
                        
                        load_time = (time.time() - start_time) * 1000
                        
                        # Test browser-specific functionality
                        browser_tests = await self.run_browser_compatibility_tests(page)
                        
                        self.results['browser_tests'].append({
                            'browser': browser_name,
                            'page': test_page['name'],
                            'load_time': load_time,
                            'tests': browser_tests,
                            'success': all(test.get('passed', False) for test in browser_tests),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        status = "✅" if all(test.get('passed', False) for test in browser_tests) else "⚠️"
                        print(f"    {status} {test_page['name']}: {load_time:.0f}ms")
                        
                    except Exception as e:
                        print(f"    ❌ {test_page['name']}: {str(e)}")
                        self.results['browser_tests'].append({
                            'browser': browser_name,
                            'page': test_page['name'],
                            'error': str(e),
                            'success': False,
                            'timestamp': datetime.now().isoformat()
                        })
                
                await context.close()
                await browser.close()

    async def run_browser_compatibility_tests(self, page):
        """Run browser-specific compatibility tests"""
        tests = []
        
        try:
            # Test 1: JavaScript functionality
            js_working = await page.evaluate("() => typeof window !== 'undefined' && typeof document !== 'undefined'")
            tests.append({
                'name': 'JavaScript Support',
                'passed': js_working,
                'details': 'Basic JavaScript and DOM access'
            })
            
            # Test 2: CSS Grid/Flexbox support
            css_modern = await page.evaluate("""
                () => {
                    const testDiv = document.createElement('div');
                    testDiv.style.display = 'grid';
                    const supportsGrid = testDiv.style.display === 'grid';
                    
                    testDiv.style.display = 'flex';
                    const supportsFlex = testDiv.style.display === 'flex';
                    
                    return supportsGrid && supportsFlex;
                }
            """)
            
            tests.append({
                'name': 'Modern CSS Support',
                'passed': css_modern,
                'details': 'CSS Grid and Flexbox support'
            })
            
            # Test 3: WebSocket support
            websocket_support = await page.evaluate("() => typeof WebSocket !== 'undefined'")
            tests.append({
                'name': 'WebSocket Support',
                'passed': websocket_support,
                'details': 'Real-time communication capability'
            })
            
            # Test 4: Local Storage support
            storage_support = await page.evaluate("""
                () => {
                    try {
                        localStorage.setItem('test', 'test');
                        localStorage.removeItem('test');
                        return true;
                    } catch (e) {
                        return false;
                    }
                }
            """)
            
            tests.append({
                'name': 'Local Storage',
                'passed': storage_support,
                'details': 'Browser storage functionality'
            })
            
            # Test 5: Media queries
            media_queries = await page.evaluate("""
                () => window.matchMedia && typeof window.matchMedia === 'function'
            """)
            
            tests.append({
                'name': 'Media Queries',
                'passed': media_queries,
                'details': 'Responsive design support'
            })
            
        except Exception as e:
            tests.append({
                'name': 'Browser Compatibility Suite',
                'passed': False,
                'error': str(e)
            })
        
        return tests

    async def test_accessibility(self):
        """Test accessibility compliance"""
        print("♿ Testing Accessibility...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            for test_page in self.test_pages:
                try:
                    await page.goto(f"{self.base_url}{test_page['path']}")
                    await page.wait_for_load_state('networkidle', timeout=10000)
                    
                    accessibility_tests = await self.run_accessibility_tests(page)
                    
                    self.results['accessibility_tests'].append({
                        'page': test_page['name'],
                        'tests': accessibility_tests,
                        'score': sum(1 for test in accessibility_tests if test.get('passed', False)),
                        'total': len(accessibility_tests),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    score = sum(1 for test in accessibility_tests if test.get('passed', False))
                    total = len(accessibility_tests)
                    print(f"    {test_page['name']}: {score}/{total} accessibility tests passed")
                    
                except Exception as e:
                    print(f"    ❌ {test_page['name']}: {str(e)}")
            
            await context.close()
            await browser.close()

    async def run_accessibility_tests(self, page):
        """Run accessibility tests on a page"""
        tests = []
        
        try:
            # Test 1: Alt text for images
            images = await page.query_selector_all('img')
            images_with_alt = 0
            
            for img in images:
                alt = await img.get_attribute('alt')
                if alt is not None:
                    images_with_alt += 1
            
            alt_text_compliance = images_with_alt == len(images) if images else True
            
            tests.append({
                'name': 'Image Alt Text',
                'passed': alt_text_compliance,
                'details': f"{images_with_alt}/{len(images)} images have alt text"
            })
            
            # Test 2: Form labels
            inputs = await page.query_selector_all('input, textarea, select')
            labeled_inputs = 0
            
            for input_elem in inputs:
                # Check for associated label
                input_id = await input_elem.get_attribute('id')
                aria_label = await input_elem.get_attribute('aria-label')
                aria_labelledby = await input_elem.get_attribute('aria-labelledby')
                
                has_label = False
                if input_id:
                    label = await page.query_selector(f'label[for="{input_id}"]')
                    has_label = label is not None
                
                if aria_label or aria_labelledby or has_label:
                    labeled_inputs += 1
            
            form_labels_compliance = labeled_inputs == len(inputs) if inputs else True
            
            tests.append({
                'name': 'Form Labels',
                'passed': form_labels_compliance,
                'details': f"{labeled_inputs}/{len(inputs)} form elements have labels"
            })
            
            # Test 3: Keyboard navigation
            focusable_elements = await page.query_selector_all(
                'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
            )
            
            keyboard_accessible = len(focusable_elements) > 0
            
            tests.append({
                'name': 'Keyboard Navigation',
                'passed': keyboard_accessible,
                'details': f"{len(focusable_elements)} focusable elements found"
            })
            
            # Test 4: Heading structure
            headings = await page.query_selector_all('h1, h2, h3, h4, h5, h6')
            heading_levels = []
            
            for heading in headings:
                tag_name = await heading.evaluate('el => el.tagName.toLowerCase()')
                level = int(tag_name[1])
                heading_levels.append(level)
            
            # Check if heading levels are logical (no skipping)
            logical_headings = True
            if heading_levels:
                for i in range(1, len(heading_levels)):
                    if heading_levels[i] > heading_levels[i-1] + 1:
                        logical_headings = False
                        break
            
            tests.append({
                'name': 'Heading Structure',
                'passed': logical_headings,
                'details': f"Heading levels: {heading_levels}"
            })
            
            # Test 5: Color contrast (basic check)
            contrast_issues = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    let issues = 0;
                    
                    for (let el of elements) {
                        if (el.textContent && el.textContent.trim()) {
                            const styles = window.getComputedStyle(el);
                            const color = styles.color;
                            const bgColor = styles.backgroundColor;
                            
                            // Basic check for very light text on light background
                            if (color === 'rgb(255, 255, 255)' && 
                                (bgColor === 'rgb(255, 255, 255)' || bgColor === 'rgba(0, 0, 0, 0)')) {
                                issues++;
                            }
                        }
                    }
                    
                    return issues;
                }
            """)
            
            tests.append({
                'name': 'Color Contrast',
                'passed': contrast_issues == 0,
                'details': f"{contrast_issues} potential contrast issues found"
            })
            
        except Exception as e:
            tests.append({
                'name': 'Accessibility Test Suite',
                'passed': False,
                'error': str(e)
            })
        
        return tests

    def generate_report(self):
        """Generate comprehensive mobile and browser compatibility report"""
        print("\n📋 Mobile & Browser Compatibility Report")
        print("=" * 60)
        
        # Mobile Responsiveness Summary
        if self.results['mobile_tests']:
            mobile_passed = [t for t in self.results['mobile_tests'] if t.get('success', False)]
            print(f"\n📱 Mobile Responsiveness:")
            print(f"  Tests passed: {len(mobile_passed)}/{len(self.results['mobile_tests'])}")
            
            # Per device summary
            devices_tested = set(t['device'] for t in self.results['mobile_tests'])
            for device in devices_tested:
                device_tests = [t for t in self.results['mobile_tests'] if t['device'] == device]
                device_passed = [t for t in device_tests if t.get('success', False)]
                print(f"  {device}: {len(device_passed)}/{len(device_tests)} pages passed")
        
        # Browser Compatibility Summary
        if self.results['browser_tests']:
            browser_passed = [t for t in self.results['browser_tests'] if t.get('success', False)]
            print(f"\n🌐 Browser Compatibility:")
            print(f"  Tests passed: {len(browser_passed)}/{len(self.results['browser_tests'])}")
            
            # Per browser summary
            browsers_tested = set(t['browser'] for t in self.results['browser_tests'])
            for browser in browsers_tested:
                browser_tests = [t for t in self.results['browser_tests'] if t['browser'] == browser]
                browser_passed = [t for t in browser_tests if t.get('success', False)]
                print(f"  {browser.title()}: {len(browser_passed)}/{len(browser_tests)} pages passed")
        
        # Accessibility Summary
        if self.results['accessibility_tests']:
            total_accessibility_score = sum(t['score'] for t in self.results['accessibility_tests'])
            total_accessibility_tests = sum(t['total'] for t in self.results['accessibility_tests'])
            
            print(f"\n♿ Accessibility:")
            print(f"  Overall score: {total_accessibility_score}/{total_accessibility_tests}")
            print(f"  Compliance rate: {total_accessibility_score/total_accessibility_tests*100:.1f}%")
        
        # Performance Summary
        all_load_times = []
        for test_type in ['mobile_tests', 'browser_tests']:
            for test in self.results[test_type]:
                if 'load_time' in test:
                    all_load_times.append(test['load_time'])
        
        if all_load_times:
            import statistics
            print(f"\n⚡ Performance:")
            print(f"  Average load time: {statistics.mean(all_load_times):.0f}ms")
            print(f"  Fastest load: {min(all_load_times):.0f}ms")
            print(f"  Slowest load: {max(all_load_times):.0f}ms")
        
        return self.results

    def save_results(self, filename=None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mobile_browser_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def main():
    """Run complete mobile and browser testing suite"""
    print("📱 AI Companion Mobile & Browser Testing")
    print("=" * 50)
    
    tester = MobileResponsivenessTester()
    
    try:
        # Run all tests
        await tester.test_mobile_responsiveness()
        await tester.test_cross_browser_compatibility()
        await tester.test_accessibility()
        
        # Generate and save report
        tester.generate_report()
        tester.save_results()
        
        print("\n✅ Mobile and browser testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Run the mobile and browser tests
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 All mobile and browser tests completed!")
    else:
        print("\n❌ Testing encountered errors")
        exit(1)
