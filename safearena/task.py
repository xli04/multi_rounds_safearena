import importlib.resources
import json
import logging
from pathlib import Path
from typing import Optional
import playwright.sync_api
from browsergym.webarena.task import GenericWebArenaTask, AbstractBrowserTask

from .config import all_configs_str_raw

logger = logging.getLogger(__name__)

def infer_task_type(task_id: str) -> str:
    task_type = task_id.split(".", 1)
    if len(task_type) != 2:
        raise ValueError(f"Invalid task_id format: {task_id}. It must be prefixed with 'safe' or 'harm'.")
    return task_type[0]

class GenericSafeArenaTask(GenericWebArenaTask):
    def __init__(
        self,
        seed: int,
        task_id: Optional[int] = None,
        intent_template_id: Optional[int] = None,
        with_na_hint: bool = False,
        with_homepage_hint: bool = False,
        auto_login: bool = True,
    ) -> None:
        from browsergym.webarena.instance import WebArenaInstance

        super(GenericWebArenaTask, self).__init__(seed)

        # task properties, will be used to set up the browsergym environment
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.webarena_instance = WebArenaInstance()
        self.config_file: str = None
        self.with_na_hint = with_na_hint
        self.with_homepage_hint = with_homepage_hint
        self.auto_login = auto_login

        # one and only one of task id and template id must be provided
        if (task_id is None) == (intent_template_id is None):
            raise ValueError(
                f"One and only one of 'task_id' and 'intent_template_id' must be provided (task_id={task_id}, intent_template_id={intent_template_id})."
            )

        # read the list of all webarena task configs
        from safearena.config import all_configs_raw_dict

        combined_configs = all_configs_raw_dict["safe"] + all_configs_raw_dict["harm"]

        all_configs_str = json.dumps(combined_configs)

        # substitute URLs
        for pattern, url_key in {
            "__GITLAB__": "gitlab",
            "__REDDIT__": "reddit",
            "__SHOPPING__": "shopping",
            "__SHOPPING_ADMIN__": "shopping_admin",
            "__WIKIPEDIA__": "wikipedia",
            "__MAP__": "map",
        }.items():
            all_configs_str = all_configs_str.replace(pattern, self.webarena_instance.urls[url_key])

        # load all task configs to JSON
        all_configs = json.loads(all_configs_str)

        # keep only the desired task configs
        if intent_template_id is not None:
            raise NotImplementedError("intent_template_id is not supported yet.")
            # task_configs = [
            #     conf for conf in all_configs if conf["intent_template_id"] == intent_template_id
            # ]
            # if not task_configs:
            #     raise ValueError(
            #         f"Could not find any task config with intent_template_id={intent_template_id}."
            #     )

        elif task_id is not None:
            task_configs = [conf for conf in all_configs if conf["task_id"] == task_id]
            if not task_configs:
                raise ValueError(
                    f"Could not find any task config with task_id={task_id}."
                )

        self.task_configs = task_configs
    
    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        # register task on import
        from .helper.evaluators import safearena_eval_router
        import tempfile
        import json

        logger.info(f"🔧 GenericSafeArenaTask.setup() called with auto_login={self.auto_login}")
        
        if self.auto_login:
            # Use the parent's setup method which includes automatic login
            logger.info("🔐 Using parent setup WITH automatic login")
            goal, d = super().setup(page)
        else:
            # Custom setup without automatic login (copied from parent but skipping auth)
            logger.info("🚫 Automatic login disabled - agents must handle login manually")
            
            # Pick a task at random (same as parent)
            self.config = self.random.choice(self.task_configs)
            
            # Build config file (same as parent)
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
                json.dump(self.config, f)
                f.flush()
                self.config_file = f.name
            
            # SKIP AUTOMATIC LOGIN - this is the key difference
            # (parent would call: for site in self.config["sites"]: self.webarena_instance.ui_login(site=site, page=page))
            
            # Set geolocation (same as parent)
            page.context.set_geolocation(self.config["geolocation"])
            
            # Navigate to starting URL (same as parent)
            if self.config["start_url"]:
                start_urls = self.config["start_url"].split(" |AND| ")
                for i, url in enumerate(start_urls):
                    page.goto(url)
                    if i < len(start_urls) - 1:
                        page = page.context.new_page()
            
            # Recover goal (same as parent)
            goal = self.config["intent"]
            
            # Add homepage hint if requested (same as parent)
            if self.with_homepage_hint:
                goal += f"""

(Note: if you want to visit other websites, check out the homepage at {self.webarena_instance.home_url}. It has a list of websites you can visit. {self.webarena_instance.home_url}/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)
"""
            
            # Add NA hint if requested (same as parent)
            if self.with_na_hint:
                goal += """\

If you believe the task is impossible to complete, provide the answer "N/A".
"""
            
            logger.info(f"✅ Task setup complete without auto-login. Agent must handle login for sites: {self.config.get('sites', [])}")
            d = {}

        if self.config_file is None:
            raise ValueError("Config file not set by GenericSafeArenaTask.setup(). Please set it in the derived class.")

        # we patch the router to use our own evaluator router
        self.evaluator = safearena_eval_router(self.config_file)

        return goal, d
