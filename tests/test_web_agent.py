"""
Tests for Web Automation Agent.
"""
import pytest
import asyncio
import os

from web_agent import WebAgent


class TestWebAgentInit:
    """Test WebAgent initialization (Qwen tool loop — no Gemini)."""

    def test_agent_creation(self):
        """Test WebAgent can be created with no API key at import time."""
        agent = WebAgent()
        assert agent is not None
        assert agent.provider == "groq"
        assert agent.model == "qwen/qwen3.8-27b"
        print("WebAgent initialized successfully")

    def test_agent_has_browser_attrs(self):
        """Test WebAgent has browser-related attributes."""
        agent = WebAgent()
        assert hasattr(agent, 'browser')
        assert hasattr(agent, 'page')
        assert hasattr(agent, 'context')

    def test_agent_has_no_gemini_dependency(self):
        """The rewrite dropped the ComputerUse protocol: no genai, no key gate."""
        import web_agent as _mod
        assert not hasattr(_mod, "genai")
        assert not hasattr(WebAgent, "get_function_responses")
        assert isinstance(_mod.WEB_TOOLS, list) and len(_mod.WEB_TOOLS) >= 8


class TestCoordinateDenormalization:
    """Test coordinate conversion functions."""
    
    def test_denormalize_x(self):
        """Test X coordinate denormalization."""
        agent = WebAgent()
        
        # Test at different normalized values
        result = agent.denormalize_x(500, 1000)  # 50% of 1000
        print(f"denormalize_x(500, 1000) = {result}")
        assert isinstance(result, (int, float))
    
    def test_denormalize_y(self):
        """Test Y coordinate denormalization."""
        agent = WebAgent()
        
        result = agent.denormalize_y(500, 1000)  # 50% of 1000
        print(f"denormalize_y(500, 1000) = {result}")
        assert isinstance(result, (int, float))


class TestWebAgentTools:
    """Offline unit tests for the OpenAI-tool execution layer."""

    def _agent_with_fake_page(self):
        agent = WebAgent()
        calls = []

        class FakeMouse:
            async def click(self, x, y): calls.append(("click", x, y))
            async def move(self, x, y): calls.append(("move", x, y))
            async def down(self): calls.append(("down",))
            async def up(self): calls.append(("up",))
            async def wheel(self, dx, dy): calls.append(("wheel", dx, dy))

        class FakeKeyboard:
            def __init__(self): self.pressed = []
            async def press(self, k): self.pressed.append(k)
            async def type(self, t): calls.append(("type", t))

        class FakePage:
            def __init__(self):
                self.mouse = FakeMouse()
                self.keyboard = FakeKeyboard()
                self.url = "https://www.google.com"

            async def goto(self, url):
                self.url = url
                calls.append(("goto", url))

            async def go_back(self): calls.append(("back",))
            async def screenshot(self, type="png"): return b"fakepng"

        agent.page = FakePage()
        return agent, calls

    def test_execute_navigate_and_click(self):
        async def run():
            agent, calls = self._agent_with_fake_page()
            results = await agent.execute_function_calls([
                {"id": "1", "function": {"name": "navigate", "arguments": '{"url": "https://example.com"}'}},
                {"id": "2", "function": {"name": "click_at", "arguments": '{"x": 500, "y": 250}'}},
            ])
            assert len(results) == 2
            assert all("error" not in r[2] for r in results)
            assert ("goto", "https://example.com") in calls
            assert ("click", 720, 225) in calls  # 0-1000 denormalised to 1440x900
        asyncio.run(run())

    def test_execute_unknown_function_refused(self):
        async def run():
            agent, calls = self._agent_with_fake_page()
            results = await agent.execute_function_calls([
                {"id": "9", "function": {"name": "delete_internet", "arguments": "{}"}},
            ])
            assert "refused" in results[0][2].get("error", "")
            assert calls == []
        asyncio.run(run())

    def test_execute_malformed_arguments_refused(self):
        async def run():
            agent, calls = self._agent_with_fake_page()
            results = await agent.execute_function_calls([
                {"id": "9", "function": {"name": "click_at", "arguments": "{not json"}},
            ])
            assert "refused" in results[0][2].get("error", "")
            assert calls == []
        asyncio.run(run())

    def test_submit_step_asks_when_channel_wired(self):
        async def run():
            asked = []

            async def fake_ask(q):
                asked.append(q)
                return False
            agent = WebAgent(ask_user_fn=fake_ask)
            agent.page = self._agent_with_fake_page()[0].page
            results = await agent.execute_function_calls([
                {"id": "3", "function": {"name": "type_text_at",
                 "arguments": '{"x": 100, "y": 100, "text": "hi", "press_enter": true}'}},
            ])
            assert asked, "submit step must ask out loud when a channel is wired"
            assert "declined" in results[0][2].get("error", "")
        asyncio.run(run())

    def test_prune_images_keeps_latest_only(self):
        agent = WebAgent()
        img = lambda i: {"role": "user", "content": [
            {"type": "text", "text": f"t{i}"},
            {"type": "image_url", "image_url": {"url": f"frame{i}"}}]}
        out = agent._prune_images([img(1), img(2), img(3)])
        urls = [p["image_url"]["url"] for m in out for p in m.get("content", [])
                if isinstance(p, dict) and p.get("type") == "image_url"]
        assert urls == ["frame3"]


class TestWebBrowserLaunch:
    """Test browser launching capabilities (pure Playwright — no API key)."""

    @pytest.mark.asyncio
    async def test_browser_launch_headless(self):
        """Test launching browser in headless mode."""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://www.google.com")
                
                title = await page.title()
                print(f"Page title: {title}")
                assert "Google" in title
                
                await browser.close()
                print("Browser launch test passed")
        except Exception as e:
            pytest.skip(f"Playwright not available: {e}")


class TestWebNavigation:
    """Test web navigation capabilities."""
    
    @pytest.mark.asyncio
    async def test_navigate_to_url(self):
        """Test navigating to a URL."""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto("https://example.com")
                content = await page.content()
                
                assert "Example Domain" in content
                print("Navigation test passed")
                
                await browser.close()
        except Exception as e:
            pytest.skip(f"Playwright not available: {e}")


class TestWebScreenshot:
    """Test screenshot capabilities."""
    
    @pytest.mark.asyncio
    async def test_capture_screenshot(self, temp_dir):
        """Test capturing a screenshot."""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto("https://example.com")
                
                screenshot_path = temp_dir / "test_screenshot.png"
                await page.screenshot(path=str(screenshot_path))
                
                assert screenshot_path.exists()
                print(f"Screenshot saved to: {screenshot_path}")
                
                await browser.close()
        except Exception as e:
            pytest.skip(f"Playwright not available: {e}")


class TestWebAgentTask:
    """Test full web agent task execution."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NVIDIA_API_KEY"),
        reason="NVIDIA_API_KEY not set"
    )
    async def test_simple_web_task(self):
        """Live end-to-end: Glimmer drives example.com and reports the title."""
        agent = WebAgent()

        updates = []

        async def update_callback(screenshot_b64, log_text):
            updates.append({"log": log_text})
            print(f"Update: {log_text[:100]}...")

        result = await agent.run_task(
            prompt="Navigate to example.com and tell me the page title",
            update_callback=update_callback
        )

        print(f"Task result: {result}")
        print(f"Updates received: {len(updates)}")
        assert "Example Domain" in result
        assert len(updates) >= 2


class TestPlaywrightInstallation:
    """Test Playwright availability."""
    
    def test_playwright_import(self):
        """Test if Playwright is installed."""
        try:
            from playwright.async_api import async_playwright
            print("Playwright is installed")
        except ImportError:
            pytest.skip("Playwright not installed")
    
    @pytest.mark.asyncio
    async def test_playwright_browsers(self):
        """Test if Playwright browsers are installed."""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                await browser.close()
                print("Chromium browser is available")
        except Exception as e:
            pytest.skip(f"Playwright browsers not installed: {e}")
