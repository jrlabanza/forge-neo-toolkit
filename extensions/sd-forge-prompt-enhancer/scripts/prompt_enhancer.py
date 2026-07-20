"""
sd-forge-prompt-enhancer
========================

Guided prompt builder for Illustrious-XL / NoobAI XL on Forge.

The user picks from human-readable dropdowns (Pose: "sitting", Shot:
"upper body", Style: "soft watercolor") and we translate each choice into
the booru tags those models were trained on. Output is assembled in
Illustrious canonical tag order:
  subject -> outfit -> pose/action -> framing -> location -> time/lighting
  -> mood -> art-style -> year/source/rating -> quality -> custom extras

Send-to-txt2img/img2img uses Forge's paste-params API.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple, Dict

import gradio as gr

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[prompt-enhancer]"

NONE = "(skip)"

# ===========================================================================
# 1. Subject — single merged dropdown so count/type can never conflict.
# ===========================================================================

SUBJECTS = {
    "Solo - 1girl (young adult female)": "1girl, solo",
    "Solo - 1boy (young adult male)":    "1boy, solo",
    "Solo - mature woman":               "1girl, solo, mature_female",
    "Solo - elderly":                    "1other, solo, old",
    "Solo - non-human creature":         "1other, solo",
    "Pair - 2girls":                     "2girls",
    "Pair - 2boys":                      "2boys",
    "Pair - 1girl + 1boy":               "1girl, 1boy",
    "Trio - 3girls":                     "3girls",
    "Trio - 3boys":                      "3boys",
    "Trio - 2girls + 1boy":              "2girls, 1boy",
    "Trio - 1girl + 2boys":              "1girl, 2boys",
    "Group - multiple girls":            "multiple_girls",
    "Group - multiple boys":             "multiple_boys",
    "Group - mixed":                     "multiple_girls, multiple_boys",
    NONE:                                "",
}

# ===========================================================================
# 2. Pose / Action
# ===========================================================================

POSES = {
    "Standing":                   "standing",
    "Standing, hands on hips":    "standing, hands_on_hips",
    "Standing, arms crossed":     "standing, crossed_arms",
    "Standing, hands behind back":"standing, hands_behind_back",
    "Standing, leaning forward":  "standing, leaning_forward",
    "Standing, hands in pockets": "standing, hand_in_pocket",
    "Walking":                    "walking",
    "Running":                    "running",
    "Sitting on chair":           "sitting, on_chair",
    "Sitting on floor":           "sitting, on_floor",
    "Sitting cross-legged":       "sitting, indian_style",
    "Sitting on knees (seiza)":   "sitting, seiza",
    "Kneeling":                   "kneeling",
    "Lying on back":              "lying, on_back",
    "Lying on stomach":           "lying, on_stomach",
    "Lying on side":              "lying, on_side",
    "Jumping":                    "jumping",
    "Falling":                    "falling",
    "Flying":                     "flying",
    "Dancing":                    "dancing",
    "Crouching":                  "squatting",
    "Hugging knees":              "knees_to_chest, hugging_own_legs",
    "Combat / fighting stance":   "fighting_stance",
    "Spell-casting":              "casting_spell, magical_aura",
    "Holding weapon":             "holding_weapon",
    "Leaning against wall":       "leaning, against_wall",
    "Stretching arms above head": "arms_up, stretching",
    "Twisting / looking back":    "twisting, looking_back",
    # --- [NSFW] poses ---
    "[NSFW] Bent over (suggestive)":     "bent_over, looking_at_viewer",
    "[NSFW] All fours / on hands & knees":"all_fours, on_hands_and_knees",
    "[NSFW] Spreading legs":             "spread_legs, presenting",
    "[NSFW] Top-down bottom-up":         "top-down_bottom-up",
    "[NSFW] Reclining on bed":           "lying, on_back, on_bed, pillow",
    "[NSFW] M-shaped legs":              "spread_legs, m_legs",
    "[NSFW] Embracing partner":          "hug, embracing",
    "[NSFW] Kissing":                    "kissing, kiss",
    "[NSFW] Cowgirl position":           "cowgirl_position",
    "[NSFW] Touching self":              "masturbation, fingering",
    "[NSFW] Squatting (revealing)":      "squatting, spread_legs",
    # --- More safe poses ---
    "Standing on tip-toes":              "standing, tiptoe",
    "Standing, one leg up":              "standing, leg_up",
    "Standing, one knee up":             "standing, knee_up",
    "Sitting at desk":                   "sitting, on_chair, desk",
    "Sitting on bed":                    "sitting, on_bed",
    "Sitting on window":                 "sitting, on_windowsill",
    "Sitting, arms back":                "sitting, arms_behind_back",
    "Crouching, hands on ground":        "crouching, hands_on_ground",
    "Hands clasped (prayer)":            "own_hands_together, praying",
    "Yoga pose":                         "yoga, balance",
    "Stretching back / arching":         "arched_back, stretching",
    "Curled up":                         "curled_up, fetal_position",
    "Star pose (arms and legs out)":     "spread_arms, spread_legs, star_pose",
    "Heroic / superhero pose":           "standing, heroic_pose, hands_on_hips",
    "Defensive / guard stance":          "fighting_stance, defensive",
    "Riding (horse / vehicle)":          "riding",
    "Skating / ice skating":             "skating, ice_skating",
    "Skateboarding":                     "skateboard, riding",
    "Bicycling":                         "bicycle, riding",
    "Surfing":                           "surfboard, surfing",
    "Diving":                            "diving, underwater",
    "Carrying someone (princess)":       "carrying, princess_carry",
    "Piggyback":                         "carrying, piggyback",
    "Holding hands with partner":        "holding_hands",
    NONE:                         "",
}

ACTIONS = {
    "Looking at viewer":     "looking_at_viewer",
    "Looking away":          "looking_away",
    "Looking back":          "looking_back",
    "Looking up":            "looking_up",
    "Looking down":          "looking_down",
    "Looking to the side":   "looking_to_the_side",
    "Closed eyes":           "closed_eyes",
    "Eyes closed, smiling":  "closed_eyes, smile",
    "Holding something":     "holding",
    "Holding cup / drink":   "holding_cup",
    "Holding phone":         "holding_phone",
    "Holding book":          "holding_book",
    "Reading book":          "reading, book",
    "Eating":                "eating",
    "Drinking":              "drinking",
    "Sleeping":              "sleeping, closed_eyes",
    "Waving":                "waving",
    "Pointing at viewer":    "pointing_at_viewer",
    "Peace sign":            "v",
    "Double peace sign":     "double_v",
    "Hand on cheek":         "hand_on_own_cheek",
    "Hand on chin":          "hand_on_own_chin",
    "Hand in hair":          "hand_in_own_hair",
    "Arms up / stretching":  "arms_up, stretching",
    "Shy / blushing":        "blush, embarrassed",
    "Crying":                "tears, crying",
    "Laughing":              "laughing, open_mouth",
    "Yawning":               "yawn, sleepy",
    "Tilting head":          "head_tilt",
    "Adjusting glasses":     "adjusting_glasses",
    "Hair flip":             "hair_flip",
    # --- [NSFW] actions ---
    "[NSFW] Lifting skirt":          "skirt_lift, panties_visible",
    "[NSFW] Pulling clothes aside":  "clothes_pull, exposing",
    "[NSFW] Covering breasts":       "covering_breasts, embarrassed",
    "[NSFW] Covering crotch":        "covering_crotch",
    "[NSFW] Squeezing own breast":   "grabbing_own_breast",
    "[NSFW] Tongue out (sexy)":      "tongue_out, seductive_smile",
    "[NSFW] Biting lip":             "biting_lip, seductive",
    "[NSFW] Undressing":             "undressing, unbuttoning",
    "[NSFW] Self-cupping panties":   "hand_in_panties",
    "[NSFW] Pulling panties down":   "panty_pull, downblouse",
    # --- More safe actions ---
    "Brushing hair":                  "brushing_hair, hairbrush",
    "Applying makeup":                "applying_makeup",
    "Tying hair":                     "tying_hair, ponytail",
    "Putting on glasses":             "adjusting_glasses, glasses",
    "Holding flower":                 "holding_flower, flower",
    "Holding sword":                  "holding_sword, sword",
    "Holding gun":                    "holding_gun, weapon",
    "Holding umbrella":               "holding_umbrella, umbrella",
    "Holding bag":                    "holding_bag, bag",
    "Cooking":                        "cooking, frying_pan",
    "Studying":                       "studying, book, pen",
    "Writing":                        "writing, pen, notebook",
    "Drawing / painting":             "drawing, painting, brush",
    "Singing":                        "singing, microphone, open_mouth",
    "Playing instrument":             "playing_instrument, music",
    "Whispering":                     "whispering, hand_to_mouth",
    "Throwing":                       "throwing, arm_back",
    "Catching":                       "catching, arms_out",
    "Cheering":                       "cheering, arms_up, open_mouth",
    "Saluting":                       "salute, formal",
    "Bowing":                         "bowing, formal",
    "Curtsy":                         "curtsy, holding_dress",
    "Pinching cheek":                 "pinching, cheek_pinching",
    "Petting cat / dog":              "petting, animal",
    NONE:                    "",
}

# ===========================================================================
# 3. Framing
# ===========================================================================

SHOT_TYPES = {
    "Portrait (head/shoulders)":     "portrait",
    "Close-up (face)":                "close-up",
    "Extreme close-up":               "extreme_close-up",
    "Upper body":                     "upper_body",
    "Cowboy shot (knees up)":         "cowboy_shot",
    "Full body":                      "full_body",
    "Wide shot (small subject)":      "wide_shot",
    "Scenery (no/tiny subject)":      "scenery",
    "POV (first person)":             "pov",
    "Selfie":                         "selfie",
    "Side profile":                   "profile, from_side",
    "Three-quarter view":             "three-quarter_view",
    "Over the shoulder":              "over_the_shoulder",
    "Establishing wide":              "scenery, very_wide_shot",
    "Macro / detail":                 "extreme_close-up, macro",
    "Action / dynamic":               "dynamic_pose, motion_lines",
    NONE:                             "",
}

CAMERA_ANGLES = {
    "Eye level (default)":  "",
    "From above":           "from_above",
    "From below":           "from_below",
    "From side":            "from_side",
    "From behind":          "from_behind",
    "Dutch angle":          "dutch_angle",
    "Bird's eye view":      "bird's-eye_view",
    "Worm's eye view":      "worm's-eye_view",
}

# ===========================================================================
# 4. Outfit
# ===========================================================================

OUTFIT_PRESETS = {
    "Casual everyday":            "casual, t-shirt, jeans",
    "Casual streetwear":          "streetwear, hoodie, jeans, sneakers",
    "School uniform (sailor)":    "school_uniform, sailor_collar, pleated_skirt",
    "School uniform (blazer)":    "school_uniform, blazer, necktie, pleated_skirt",
    "Office / business":          "business_suit, pencil_skirt, blouse",
    "Office (male)":              "business_suit, necktie, dress_shirt",
    "Formal dress":               "evening_gown, long_dress",
    "Wedding":                    "wedding_dress, veil",
    "Kimono":                     "kimono",
    "Yukata (summer festival)":   "yukata, obi",
    "Hanbok":                     "hanbok",
    "Cheongsam / qipao":          "china_dress",
    "Maid":                       "maid, maid_apron, maid_headdress",
    "Nurse":                      "nurse, nurse_cap",
    "Waitress":                   "waitress, apron",
    "Police officer":             "police_uniform, peaked_cap",
    "Soldier / military":         "military_uniform, beret",
    "Fantasy armor":              "armor, plate_armor, gauntlets",
    "Fantasy mage robes":         "wizard_robe, hood_up, witch_hat",
    "Cyberpunk":                  "cyberpunk, leather_jacket, neon_trim",
    "Sci-fi bodysuit":            "bodysuit, sci-fi, futuristic",
    "Magical girl":               "magical_girl, frills, hair_ribbon, gloves",
    "Idol stage costume":         "idol, frilled_dress, hair_bow, gloves",
    "Goth / lolita":              "gothic_lolita, frills, black_dress",
    "Sportswear (gym)":           "sportswear, sports_bra, shorts",
    "Sportswear (track)":         "track_suit, jacket",
    "Swimsuit (bikini)":          "bikini",
    "Swimsuit (one-piece)":       "one-piece_swimsuit",
    "Pajamas":                    "pajamas, long_sleeves",
    "Bathrobe / towel":           "bathrobe",
    "Winter coat":                "winter_clothes, coat, scarf",
    "Summer dress":               "summer_dress, sundress",
    "Hoodie + shorts":            "hoodie, shorts",
    "Naked (no clothes)":         "nude",
    # --- Additional safe outfits ---
    "Oversized shirt":            "oversized_shirt, off-shoulder",
    "Tank top + shorts":          "tank_top, shorts",
    "Knitted sweater":            "sweater, long_sleeves",
    "Hoodie + thighhighs":        "hoodie, thighhighs",
    "Dress shirt only":           "dress_shirt, no_pants",
    "Bodysuit (sport)":           "bodysuit, sportswear",
    "Tracksuit":                  "tracksuit, jacket",
    "Sari":                       "sari, indian_clothes",
    "Bridal lingerie":            "bridal_lingerie, lace, veil",
    # --- [NSFW] options ---
    "[NSFW] Lingerie (basic)":           "lingerie, bra, panties",
    "[NSFW] Lingerie (lacy + garter)":   "lingerie, lace, garter_belt, thighhighs",
    "[NSFW] Playboy bunny suit":         "playboy_bunny, bunny_ears, fishnet_pantyhose",
    "[NSFW] Micro bikini":               "micro_bikini, very_revealing",
    "[NSFW] Sling bikini":               "sling_bikini",
    "[NSFW] Naked apron":                "naked_apron, kitchen_apron",
    "[NSFW] Wet shirt (see-through)":    "wet_clothes, see-through, transparent",
    "[NSFW] Open shirt":                 "open_shirt, cleavage, exposed_breasts",
    "[NSFW] Crop top + miniskirt":       "crop_top, miniskirt, midriff",
    "[NSFW] Stockings + garter only":    "stockings, garter_belt, thighhighs, nude",
    "[NSFW] Topless":                    "topless, bare_breasts",
    "[NSFW] Bottomless":                 "bottomless, no_panties",
    "[NSFW] Nude + jewelry only":        "completely_nude, jewelry, necklace",
    "[NSFW] Revealing maid":             "maid, revealing_clothes, garter_belt, cleavage",
    "[NSFW] Gym uniform / bloomers":     "gym_uniform, bloomers",
    "[NSFW] Schoolgirl (undressing)":    "school_uniform, undressing, unbuttoned_shirt",
    "[NSFW] Torn clothes":               "torn_clothes, torn_shirt, exposed_skin",
    "[NSFW] Latex catsuit":              "latex, bodysuit, shiny_clothes",
    "[NSFW] Bondage outfit":             "bondage_outfit, harness, leather",
    "[NSFW] Cum on clothes":             "cum_on_clothes, dirty_clothes",
    # --- More fashion / themed outfits ---
    "Steampunk":                  "steampunk, brown_clothing, goggles",
    "Punk (leather + chains)":    "punk, leather_jacket, spiked_collar",
    "Y2K fashion":                "y2k, low_rise_pants, crop_top",
    "Cottagecore":                "cottagecore, floral_dress, sun_hat",
    "Dark academia":              "dark_academia, blazer, plaid_skirt",
    "Mori girl":                  "mori_girl, layered_clothes, earth_tones",
    "Lolita (sweet)":             "sweet_lolita, frills, bonnet, pink_dress",
    "Lolita (classic)":           "classic_lolita, lace, modest_dress",
    "Athleisure":                 "athleisure, leggings, sports_bra",
    "Sundress + sun hat":         "sundress, sun_hat, summer_outfit",
    "Trench coat":                "trench_coat, belt, long_coat",
    "Leather jacket + jeans":     "leather_jacket, jeans, casual_cool",
    "Cowgirl":                    "cowboy_hat, denim, cowgirl",
    "Pirate":                     "pirate, tricorne, eyepatch, sword",
    "Ninja":                      "ninja, mask, dark_clothes, weapon",
    "Samurai":                    "samurai_armor, katana, hakama",
    "Greek goddess":              "toga, gold_jewelry, sandals",
    "Egyptian":                   "egyptian_clothes, gold_jewelry, eye_makeup",
    "Holiday: Santa":             "santa_costume, santa_hat, red_clothes",
    "Holiday: bunny":             "bunny_ears, bunny_tail, easter",
    "Holiday: witch":             "witch_hat, witch_robe, halloween",
    "Holiday: angel":             "angel_wings, white_dress, halo",
    "Holiday: devil":             "demon_horns, demon_tail, red_clothes",
    "Cosplay (general)":          "cosplay, costume",
    "Hospital gown":              "hospital_gown, medical",
    "Apron only (cooking)":       "apron, casual",
    "Oversized hoodie only":      "hoodie, no_pants, oversized_clothes",
    NONE:                         "",
}

# ===========================================================================
# 5. Scene / Location
# ===========================================================================

LOCATIONS = {
    "Simple white background":   "white_background, simple_background",
    "Simple black background":   "black_background, simple_background",
    "Gradient background":       "gradient_background, simple_background",
    "Bedroom":                   "bedroom, indoors",
    "Living room":               "living_room, indoors, sofa",
    "Kitchen":                   "kitchen, indoors",
    "Bathroom":                  "bathroom, indoors, tile_wall",
    "Classroom":                 "classroom, indoors, school_desk",
    "School hallway":            "hallway, indoors, lockers",
    "Library":                   "library, indoors, bookshelf",
    "Cafe / coffee shop":        "cafe, indoors, table",
    "Restaurant":                "restaurant, indoors, table",
    "Office":                    "office, indoors, desk, computer",
    "Shop / store":              "shop, indoors, shelves",
    "Bar / nightclub":           "bar_(place), indoors, neon_lights",
    "Train interior":            "train_interior, indoors",
    "Park (outdoor)":            "park, outdoors, tree, grass",
    "Forest":                    "forest, outdoors, tree",
    "Beach":                     "beach, outdoors, ocean, sand",
    "Mountain landscape":        "mountain, outdoors, sky",
    "City street":               "street, outdoors, building, city",
    "Rooftop":                   "rooftop, outdoors, city",
    "School courtyard":          "courtyard, outdoors, school",
    "Garden":                    "garden, outdoors, flower",
    "Snowy outdoors":            "snow, outdoors, winter",
    "Rainy street":              "rain, outdoors, wet, reflection",
    "Cherry blossom":            "outdoors, cherry_blossoms, petals",
    "Festival night":            "festival, night, paper_lantern, outdoors",
    "Stage / concert":           "stage, indoors, spotlight, audience",
    "Castle interior":           "castle, indoors, stone_wall",
    "Dungeon":                   "dungeon, indoors, dark, torch",
    "Magical forest":            "fantasy, forest, glowing, mystical",
    "Cyberpunk city":            "cyberpunk, city, neon, night",
    "Space station":             "spaceship_interior, sci-fi",
    "Starry sky":                "sky, night_sky, stars, outdoors",
    "Pool / poolside":           "pool, poolside, water",
    "Underwater":                "underwater, bubbles",
    "Rooftop garden":            "rooftop, garden, outdoors, plants",
    "Train platform":            "train_station, platform, outdoors",
    "Locker room":               "locker_room, indoors",
    # --- Intimate locations ---
    "Bedroom (bed close-up)":    "bedroom, on_bed, sheets, pillow, indoors",
    "Hot spring / onsen":        "onsen, hot_spring, steam, water",
    "Shower":                    "shower, bathroom, water, wet",
    "Bathtub / bath":            "bathtub, bath, indoors, water",
    "[NSFW] Love hotel":         "love_hotel, indoors, neon",
    "[NSFW] Massage parlor":     "massage_parlor, indoors, oil",
    # --- More locations ---
    "Bookstore":                 "bookstore, indoors, bookshelf",
    "Toy store":                 "toy_store, indoors",
    "Convenience store":         "convenience_store, indoors",
    "Movie theater":             "movie_theater, indoors, dark",
    "Arcade":                    "arcade, indoors, neon_lights",
    "Bowling alley":             "bowling_alley, indoors",
    "Karaoke room":              "karaoke, indoors, microphone, neon",
    "Bus stop":                  "bus_stop, outdoors",
    "Subway / train interior":   "train_interior, indoors, commute",
    "Airport":                   "airport, indoors",
    "Hotel room":                "hotel_room, indoors, bed",
    "Hotel lobby":               "hotel_lobby, indoors",
    "Bridge over water":         "bridge, outdoors, water",
    "Pier / dock":               "pier, outdoors, ocean",
    "Waterfall":                 "waterfall, outdoors, nature",
    "Cliff edge":                "cliff, outdoors, sky",
    "Desert":                    "desert, outdoors, sand, sky",
    "Tundra / snowy plain":      "tundra, snow, outdoors",
    "Jungle":                    "jungle, outdoors, green, dense_foliage",
    "Tropical island":           "island, tropical, palm_tree, outdoors",
    "Cave":                      "cave, indoors, dark, rocks",
    "Aquarium":                  "aquarium, indoors, fish, blue_lighting",
    "Theme park / carnival":     "amusement_park, outdoors, ferris_wheel",
    "Ice skating rink":          "ice_rink, indoors, ice",
    "Greenhouse":                "greenhouse, indoors, plants, glass_ceiling",
    "Vineyard":                  "vineyard, outdoors, grape_vine",
    "Farm / barn":               "farm, outdoors, hay, barn",
    "Train platform":            "train_station, platform, outdoors",
    "Skyscraper office":         "office, indoors, large_windows, cityscape",
    "Penthouse":                 "penthouse, indoors, modern, view",
    "Underground / sewer":       "sewer, indoors, dark, pipes",
    "Spaceship cockpit":         "spaceship_interior, cockpit, sci-fi, consoles",
    "Throne room":               "throne_room, indoors, royal",
    "Library, antique":          "library, indoors, antique, bookshelf",
    "Train, scenic route":       "train_interior, indoors, window_view",
    NONE:                        "",
}

# ===========================================================================
# 6. Time of day + Lighting
# ===========================================================================

TIME_OF_DAY = {
    "Morning":          "morning, sunlight",
    "Noon / daytime":   "day, daylight",
    "Afternoon":        "afternoon, sunlight",
    "Golden hour":      "golden_hour, sunlight, warm_lighting",
    "Sunset":           "sunset, orange_sky",
    "Twilight":         "twilight, evening",
    "Night":            "night, dark",
    "Late night":       "night, moonlight",
    "Dawn":             "dawn, soft_lighting",
    NONE:               "",
}

LIGHTING = {
    "Soft natural":           "soft_lighting, natural_lighting",
    "Bright / sunny":         "bright, sunny",
    "Cinematic":              "cinematic_lighting, dramatic_lighting",
    "Backlit":                "backlighting, rim_lighting",
    "Rim light":              "rim_lighting",
    "Neon glow":              "neon_lights, glowing",
    "Candle / firelight":     "firelight, warm_lighting",
    "Studio lighting":        "studio_lighting",
    "Volumetric / god rays":  "volumetric_lighting, light_rays",
    "Moody / low light":      "low_light, moody, shadows",
    "Spotlight (single)":     "spotlight, single_light, dark_background",
    "Side lighting":          "side_lighting, half_shadow",
    "Top lighting (dramatic)":"top_lighting, dramatic_shadows",
    "Underwater glow":        "underwater_lighting, caustics, blue_tint",
    "Christmas / fairy lights":"fairy_lights, warm_lighting, bokeh",
    "Lantern light":          "lantern, warm_lighting, japanese",
    "Fluorescent (cold)":     "fluorescent_lighting, cold, sterile",
    "Sunset glow":            "warm_lighting, sunset_glow, orange",
    "Aurora borealis":        "aurora, northern_lights, magical",
    "Lightning strike":       "lightning, thunderstorm, dramatic",
    NONE:                     "",
}

# ===========================================================================
# 7. Mood / Expression
# ===========================================================================

EXPRESSIONS = {
    "Smiling":              "smile",
    "Big grin":             "grin, open_mouth",
    "Soft smile":           "smile, soft_expression",
    "Neutral":              "expressionless",
    "Serious":              "serious",
    "Sad":                  "sad, frown",
    "Crying":               "tears, crying",
    "Surprised":            "surprised, open_mouth",
    "Angry":                "angry",
    "Annoyed":              "annoyed",
    "Smug":                 "smug, smirk",
    "Embarrassed / blush":  "blush, embarrassed",
    "Shy":                  "shy, blush",
    "Sleepy":               "sleepy, half-closed_eyes",
    "Excited":              "excited, open_mouth",
    "Bored":                "bored",
    "Confident":            "confident",
    "Determined":           "determined",
    "Seductive":            "seductive_smile",
    "Yandere":              "yandere, crazed_eyes",
    "Smirk":                "smirk",
    "Pout":                 "pout, pouty_lips",
    "Wink":                 "wink, one_eye_closed",
    "Smiling, eyes closed": "smile, closed_eyes",
    # --- [NSFW] expressions ---
    "[NSFW] Ahegao":                "ahegao, rolling_eyes, tongue_out",
    "[NSFW] Erotic blush":          "blush, lustful_eyes, parted_lips",
    "[NSFW] Moaning":               "open_mouth, moaning, blush",
    "[NSFW] Half-lidded (sultry)":  "half-closed_eyes, sultry, bedroom_eyes",
    "[NSFW] Crying (pleasure)":     "tears, blush, open_mouth, pleasure",
    "[NSFW] Drooling":              "drooling, blush, open_mouth",
    "[NSFW] Naughty smile":         "evil_smile, smirk, lustful",
    "[NSFW] Orgasm face":           "orgasm_face, ahegao, rolling_eyes",
    # --- More expressions ---
    "Sticking tongue out (playful)":"tongue_out, playful",
    "Heart eyes":                    "heart-shaped_pupils, hearts, love",
    "Star eyes":                     "+_+, star-shaped_pupils, excited",
    "Eyes wide with awe":            "starry_eyes, wide_eyed, awe",
    "Hopeful":                       "hopeful, sparkling_eyes",
    "Worried":                       "worried, furrowed_brow",
    "Suspicious":                    "suspicious, narrowed_eyes",
    "Grumpy":                        "grumpy, pout, frown",
    "Content / peaceful":            "content_smile, peaceful",
    "Flirty":                        "flirty, half-closed_eyes",
    "Concentrating":                 "concentrated, focused, serious",
    "Daydreaming":                   "dreamy, looking_up, soft_expression",
    "Mysterious":                    "mysterious, half_smile",
    "Cocky / smug grin":             "smug, grin, raised_eyebrow",
    NONE:                   "",
}

# ===========================================================================
# 8. Art Style — model-aware
# ===========================================================================

ART_STYLES_ILLUSTRIOUS = {
    "Detailed anime (default)":    "anime_style, detailed, sharp_focus",
    "Soft pastel anime":           "pastel_colors, soft_lighting, soft_anime_style",
    "Vibrant anime":               "vibrant_colors, saturated, anime_style",
    "Cinematic / film":            "cinematic, depth_of_field, film_grain, dramatic_lighting",
    "Watercolor":                  "watercolor_(medium), traditional_media, soft_brushstrokes",
    "Oil painting":                "oil_painting_(medium), painterly, traditional_media",
    "Pencil sketch":               "sketch, pencil_drawing, monochrome, traditional_media",
    "Lineart only":                "lineart, monochrome, simple_shading",
    "Cel-shaded":                  "cel_shading, flat_colors, bold_outlines",
    "Manga (B&W)":                 "monochrome, manga, screentones, comic",
    "Painterly fantasy":           "painterly, fantasy_art, detailed_background",
    "3D / CG render":              "3d, cg, render, glossy",
    "Realistic / semi-real":       "semi-realistic, realistic, detailed_skin",
    "Chibi / SD":                  "chibi, sd_character, cute",
    "Gothic / dark":               "dark, gothic, moody, dramatic_shadows",
    "Retro 90s anime":             "1990s_(style), retro_anime, vhs_aesthetic",
    "Vaporwave":                   "vaporwave, pink_and_purple, retro_aesthetic",
    "Storybook / children's book": "storybook_illustration, soft_lines, warm_colors",
    "Studio Ghibli-esque":         "ghibli_inspired, soft_painted_background, warm_colors",
    "Konachan / wallpaper":        "wallpaper, scenic, detailed_background, official_art",
    "Western comic book":          "comic_book_style, bold_outlines, halftone",
    "Disney-esque":                "disney_style, smooth, expressive_face",
    "Pixar-esque":                 "3d, pixar_style, smooth, cute",
    "Manga ink wash":              "ink_wash, manga, monochrome",
    "Acrylic painting":            "acrylic_painting, textured, traditional",
    "Pop art":                     "pop_art, bold_colors, halftone_dots",
    "Art nouveau":                 "art_nouveau, ornate, decorative_frame",
    "Art deco":                    "art_deco, geometric, gold_accents",
    "Impressionist":               "impressionist_style, painterly, soft_focus",
    "Ukiyo-e (Japanese woodblock)":"ukiyo-e, japanese_traditional, flat_colors",
    "Surrealist":                  "surreal, dreamlike, abstract",
    "Pixel art":                   "pixel_art, retro_game",
    NONE:                          "",
}

ART_STYLES_NOOBAI = dict(ART_STYLES_ILLUSTRIOUS)

# ===========================================================================
# Artist styles (model-aware, curated)
# ===========================================================================
# Each entry: label -> (artist_tag, short_description, danbooru_url)
# Labels are prefixed with a [Category] tag so users can scan by mood.
# Selecting an artist emits either `artist_tag` (full strength) or
# `(artist_tag:strength)` if the user dials the strength slider away from 1.0.

ARTIST_STYLES = {
    "(none - use only my visual style preset)": (
        "", "No artist tag added. Output uses only the Visual style preset.", ""
    ),

    # --- Modern / clean anime ---
    "[Modern] kantoku - polished light-novel illustration": (
        "kantoku",
        "Clean modern anime, polished shading, light-novel illustrator aesthetic. "
        "Works great for character portraits and slice-of-life scenes.",
        "https://danbooru.donmai.us/posts?tags=kantoku",
    ),
    "[Modern] hiten_(hitenkei) - clean anime, soft palette": (
        "hiten_(hitenkei)",
        "Clean lines, soft palette, modern anime style. Great for cute "
        "everyday scenes.",
        "https://danbooru.donmai.us/posts?tags=hiten_(hitenkei)",
    ),
    "[Modern] ningen_mame - polished character art": (
        "ningen_mame",
        "Detailed character art with crisp shading and clean rendering. "
        "Strong faces, balanced compositions.",
        "https://danbooru.donmai.us/posts?tags=ningen_mame",
    ),
    "[Modern] ask_(askzy) - clean modern anime": (
        "ask_(askzy)",
        "Smooth modern anime, polished skin, soft lighting. Reliable allrounder.",
        "https://danbooru.donmai.us/posts?tags=ask_(askzy)",
    ),
    "[Modern] agawa_ryou - clean polished anime": (
        "agawa_ryou",
        "Clean anime aesthetic with polished lighting and balanced colors.",
        "https://danbooru.donmai.us/posts?tags=agawa_ryou",
    ),

    # --- Soft / pastel / dreamy ---
    "[Soft] ciloranko - soft pastel anime, dreamy": (
        "ciloranko",
        "Dreamy pastel colors, soft lighting, gentle expressions. "
        "Perfect for comfy / wholesome / romantic scenes.",
        "https://danbooru.donmai.us/posts?tags=ciloranko",
    ),
    "[Soft] mochizuki_kei - soft pastel": (
        "mochizuki_kei",
        "Soft pastel palette, dreamy atmosphere, gentle character faces.",
        "https://danbooru.donmai.us/posts?tags=mochizuki_kei",
    ),
    "[Soft] shal.e - cute soft anime": (
        "shal.e",
        "Cute anime aesthetic with soft colors and rounded shapes.",
        "https://danbooru.donmai.us/posts?tags=shal.e",
    ),
    "[Soft] as109 - smooth painterly soft": (
        "as109",
        "Smooth painterly anime with soft brushwork, dreamy mood.",
        "https://danbooru.donmai.us/posts?tags=as109",
    ),
    "[Soft] nardack - soft polished portraits": (
        "nardack",
        "Soft polished anime portraits with delicate skin and warm tones.",
        "https://danbooru.donmai.us/posts?tags=nardack",
    ),

    # --- Cinematic / dark / dramatic ---
    "[Cinematic] wlop - painterly dark fantasy": (
        "wlop",
        "Painterly, dramatic lighting, somber atmospheres. Highly detailed faces, "
        "moody dark backgrounds, signature ethereal feel.",
        "https://danbooru.donmai.us/posts?tags=wlop",
    ),
    "[Cinematic] ke-ta - cinematic anime, dramatic": (
        "ke-ta",
        "Cinematic anime with strong lighting contrast and dramatic compositions.",
        "https://danbooru.donmai.us/posts?tags=ke-ta",
    ),
    "[Cinematic] guweiz - detailed fantasy atmospheric": (
        "guweiz",
        "Detailed fantasy art with atmospheric lighting, deep colors. "
        "Strong for fantasy adventurers and dramatic portraits.",
        "https://danbooru.donmai.us/posts?tags=guweiz",
    ),
    "[Cinematic] nakkar - cinematic painterly": (
        "nakkar",
        "Cinematic painterly anime with strong mood lighting.",
        "https://danbooru.donmai.us/posts?tags=nakkar",
    ),
    "[Cinematic] redum4 - detailed painterly portraits": (
        "redum4",
        "Detailed painterly character art with rich textures and depth.",
        "https://danbooru.donmai.us/posts?tags=redum4",
    ),

    # --- Vibrant / pop / neon ---
    "[Vibrant] mika_pikazo - vibrant neon pop": (
        "mika_pikazo",
        "Vibrant saturated colors, neon-pop aesthetic, modern J-pop / idol feel.",
        "https://danbooru.donmai.us/posts?tags=mika_pikazo",
    ),
    "[Vibrant] redjuice - cyberpunk neon detailed": (
        "redjuice",
        "Cyberpunk-tinged detailed art with neon palette and modern UI feel.",
        "https://danbooru.donmai.us/posts?tags=redjuice",
    ),
    "[Vibrant] jaco - flat colors, clean cute": (
        "jaco",
        "Flat-color clean anime, cute compositions, easy on the eye.",
        "https://danbooru.donmai.us/posts?tags=jaco",
    ),

    # --- Watercolor / traditional ---
    "[Traditional] yoneyama_mai - watercolor traditional": (
        "yoneyama_mai",
        "Traditional watercolor-style anime with soft brushstrokes and "
        "muted natural palette.",
        "https://danbooru.donmai.us/posts?tags=yoneyama_mai",
    ),
    "[Traditional] aoi_ogata - soft watercolor": (
        "aoi_ogata",
        "Soft watercolor traditional style with delicate color washes.",
        "https://danbooru.donmai.us/posts?tags=aoi_ogata",
    ),

    # --- Retro / vintage ---
    "[Retro] lack - retro vintage anime": (
        "lack",
        "Retro vintage anime aesthetic, slightly grainy palette, 1990s flavor.",
        "https://danbooru.donmai.us/posts?tags=lack",
    ),
    "[Retro] bkub - simple cute retro": (
        "bkub_(bkub.exe)",
        "Simple cute retro style, bold outlines, flat colors. Comedic flavor.",
        "https://danbooru.donmai.us/posts?tags=bkub_(bkub.exe)",
    ),

    # --- Painterly / fantasy ---
    "[Painterly] huanxiang_heitu - Chinese painterly": (
        "huanxiang_heitu",
        "Chinese painterly anime with rich detail and atmospheric environments.",
        "https://danbooru.donmai.us/posts?tags=huanxiang_heitu",
    ),
    "[Painterly] kawacy - illustrative painterly": (
        "kawacy",
        "Illustrative painterly anime with dramatic compositions and warm colors.",
        "https://danbooru.donmai.us/posts?tags=kawacy",
    ),
}


def _load_full_artist_list():
    """Load the 33k Illustrious-NoobAI compatible artist list bundled with
    this extension. Names come space-separated with escaped parens; we
    normalize to booru form (spaces -> underscores, unescape parens) for
    emission, while keeping the human-readable name for the dropdown label.
    Returns a list of (display_label, booru_tag) tuples and a {label: tag}
    dict for fast lookup.
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "artists.txt")
    items = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                name = raw.strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                # Booru tag: spaces -> underscores, unescape backslash-parens
                tag = name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_")
                items.append((name, tag))
    except FileNotFoundError:
        items = []
    return items


# Loaded once at import.
FULL_ARTIST_ITEMS = _load_full_artist_list()
FULL_ARTIST_LABEL_TO_TAG = {label: tag for label, tag in FULL_ARTIST_ITEMS}
FULL_ARTIST_LABELS = ["(none)"] + [lbl for lbl, _ in FULL_ARTIST_ITEMS]
logger.info("{} loaded {} full-list artists".format(TAG, len(FULL_ARTIST_ITEMS)))


def _danbooru_preview_url(booru_tag, rating="g,s"):
    """Server-side fetch of one preview image URL from Danbooru. Returns the
    image URL string, or None on any failure. Anonymous endpoint, no auth."""
    if not booru_tag:
        return None
    try:
        import urllib.request, urllib.parse, json, ssl
        q = urllib.parse.quote(booru_tag)
        url = "https://danbooru.donmai.us/posts.json?tags={}&limit=1&random=true".format(q)
        req = urllib.request.Request(
            url, headers={"User-Agent": "sd-forge-prompt-enhancer/1.0"}
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=6, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None
        post = data[0] if isinstance(data, list) else data
        # Prefer the smaller preview, fall back to sample/file
        return (post.get("preview_file_url")
                or post.get("large_file_url")
                or post.get("file_url"))
    except Exception as e:
        logger.warning("{} danbooru preview fetch failed: {}".format(TAG, e))
        return None


# ===========================================================================
# WD14 image tagger (lazy-loaded)
# ===========================================================================
# SmilingWolf's WD14 ViT v3 tagger is the same family of model Illustrious /
# NoobAI were trained against, so its output drops straight into an
# Illustrious-format prompt for near 1:1 replication.

# WD14 model registry. All v3 models use 448x448 input.
WD14_MODELS = {
    "Fast (ViT base, ~360MB)": {
        "repo":   "SmilingWolf/wd-vit-tagger-v3",
        "subdir": "vit_v3",
    },
    "Balanced (ViT large, ~1.1GB)": {
        "repo":   "SmilingWolf/wd-vit-large-tagger-v3",
        "subdir": "vit_large_v3",
    },
    "Best (EVA02 large, ~1.3GB - default for serious style detection)": {
        "repo":   "SmilingWolf/wd-eva02-large-tagger-v3",
        "subdir": "eva02_large_v3",
    },
}
WD14_DEFAULT_MODEL_KEY = "Best (EVA02 large, ~1.3GB - default for serious style detection)"
WD14_TARGET_SIZE = 448

# Cached sessions keyed by model key -> (session, tags).
_WD14_CACHE = {}

# Style descriptor tags that live in WD14 category 0 (general) but actually
# describe the art style / medium. We pull these out into their own field so
# the user can see "this looks like a watercolor in a soft-pastel style" at a
# glance. Tags are matched case-insensitively against the WD14 output name.
# Scene-context tags pulled from the general bucket for the dedicated
# "Scene context" output field. Each set groups thematically related Danbooru
# tags so we can show the user a structured breakdown of when/where/weather/
# lighting/season the image is set in. Match is case-insensitive.

SCENE_TIME_TAGS = {
    "morning", "daytime", "noon", "afternoon", "evening",
    "sunset", "sunrise", "twilight", "dusk", "dawn",
    "night", "midnight", "late_night",
    "golden_hour", "blue_hour", "magic_hour",
}
SCENE_WEATHER_TAGS = {
    "rain", "raining", "rainy", "drizzle",
    "snow", "snowing", "snowflakes", "snowstorm",
    "cloud", "clouds", "cloudy", "overcast",
    "sunny", "clear_sky",
    "fog", "foggy", "mist", "misty", "haze",
    "storm", "stormy", "thunderstorm", "lightning",
    "wind", "windy", "blizzard", "rainbow",
    "hail", "sleet",
}
SCENE_LIGHTING_TAGS = {
    "backlighting", "rim_lighting", "rim_light",
    "dramatic_lighting", "cinematic_lighting", "soft_lighting",
    "volumetric_lighting", "studio_lighting", "natural_lighting",
    "low_light", "moody", "atmospheric_lighting",
    "bright", "dark", "harsh_lighting",
    "warm_lighting", "cool_lighting", "neon_lights",
    "candlelight", "firelight", "lantern", "lanterns",
    "spotlight", "side_lighting", "top_lighting",
    "ambient_light", "key_light", "fill_light",
    "god_rays", "light_rays", "sunbeam", "sunbeams",
}
SCENE_LOCATION_TAGS = {
    "outdoors", "indoors",
    "forest", "woods", "jungle", "tree", "trees",
    "beach", "ocean", "sea", "shore", "sand",
    "mountain", "mountains", "hill", "hills", "valley", "canyon",
    "city", "cityscape", "skyline", "street", "alley", "rooftop",
    "cafe", "coffee_shop", "restaurant", "diner", "bar",
    "classroom", "school", "school_hallway", "schoolyard", "gym",
    "bedroom", "kitchen", "bathroom", "living_room", "dining_room",
    "office", "library", "bookstore", "store", "shop", "convenience_store",
    "train", "train_interior", "subway", "car_interior", "airplane",
    "shrine", "temple", "church", "cathedral", "castle", "dungeon",
    "cave", "tunnel", "ruins",
    "desert", "tundra", "field", "meadow", "garden", "park",
    "river", "lake", "pond", "waterfall", "bridge", "pier",
    "tower", "skyscraper", "balcony", "stairs", "hallway", "corridor",
    "cyberpunk", "futuristic", "fantasy", "space", "spaceship_interior",
    "stage", "concert", "festival", "amusement_park", "carnival",
    "swimming_pool", "pool", "onsen", "hot_spring", "bathhouse",
}
SCENE_SEASON_TAGS = {
    "spring", "summer", "autumn", "fall", "winter",
    "cherry_blossoms", "cherry_blossom", "sakura",
    "autumn_leaves", "fallen_leaves", "maple_leaf",
    "snowflakes",
    "flowers", "flower_field",
}
SCENE_SKY_TAGS = {
    "sky", "blue_sky", "cloudy_sky", "starry_sky", "night_sky",
    "stars", "star_(sky)", "milky_way", "galaxy",
    "moon", "full_moon", "crescent_moon", "moonlight",
    "sun", "sunlight", "aurora", "northern_lights",
    "stratosphere", "cloud_sea",
}

WD14_STYLE_TAGS = {
    # Medium
    "watercolor", "watercolor_(medium)", "watercolor_painting",
    "oil_painting", "oil_painting_(medium)", "acrylic_painting",
    "ink_wash", "ink_wash_painting", "ink_drawing",
    "pencil_drawing", "graphite_(medium)", "colored_pencil",
    "marker_(medium)", "pastel_(medium)",
    "digital_painting", "digital_media", "traditional_media",
    "3d", "3d_(artwork)", "cg", "cgi", "render", "3d_render",
    "pixel_art", "vector_art", "vector",
    # Rendering style
    "sketch", "lineart", "line_art", "rough_sketch",
    "monochrome", "greyscale", "limited_palette",
    "screentones", "halftone", "manga",
    "cel_shading", "cel_shaded", "flat_color", "flat_colors", "flat_shading",
    "painterly", "painterly_style", "brushstrokes",
    "thick_outlines", "bold_outlines", "no_outlines",
    "highly_detailed", "detailed", "intricate_details",
    "simple_background_(style)", "minimalist",
    # Color/palette feel
    "pastel_colors", "vibrant_colors", "saturated", "muted_colors",
    "warm_colors", "cool_colors", "pastel_palette",
    "high_contrast", "low_contrast", "soft_lighting", "dramatic_lighting",
    # Genre/style
    "anime_style", "anime_coloring", "anime_screencap",
    "realistic", "semi-realistic", "photorealistic", "hyperrealistic",
    "chibi", "sd_character", "deformed",
    "cartoon", "western_comic_book",
    "official_art", "fan_art", "doujinshi",
    # Texture / effect
    "film_grain", "vhs_aesthetic", "vintage",
    "blurry", "blurry_background", "depth_of_field", "bokeh",
    "lens_flare", "chromatic_aberration",
    "glitch", "glow", "shiny",
}


def _wd14_model_dir():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "wd14_model")
    os.makedirs(d, exist_ok=True)
    return d


def _wd14_download(model_key, progress_cb=None):
    """Download the WD14 ONNX model + tag list to a per-model subdir.
    Idempotent - skips if files already exist."""
    import os, urllib.request
    info = WD14_MODELS.get(model_key)
    if not info:
        raise ValueError("Unknown WD14 model: {}".format(model_key))
    base = _wd14_model_dir()
    d = os.path.join(base, info["subdir"])
    os.makedirs(d, exist_ok=True)
    model_p = os.path.join(d, "model.onnx")
    tags_p  = os.path.join(d, "selected_tags.csv")
    model_url = "https://huggingface.co/{}/resolve/main/model.onnx".format(info["repo"])
    tags_url  = "https://huggingface.co/{}/resolve/main/selected_tags.csv".format(info["repo"])

    def _dl(url, dest, label):
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            return
        if progress_cb: progress_cb("Downloading {}...".format(label))
        logger.info("{} downloading {} -> {}".format(TAG, url, dest))
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)

    _dl(model_url, model_p, "{} model".format(model_key))
    _dl(tags_url,  tags_p,  "{} tag list".format(model_key))
    return model_p, tags_p


def _wd14_load(model_key, progress_cb=None):
    """Lazy-build and cache the ONNX session + parsed tag table for the
    given model key. Subsequent calls with the same key return the cached
    session immediately.

    The cache is bounded to ONE entry (LRU=1). Switching models releases the
    previous session — the EVA02-large model alone is ~1.3GB on CUDA, and
    holding multiple of them concurrently can starve the SD checkpoint
    of GPU memory on 8-12GB cards."""
    if model_key in _WD14_CACHE:
        return _WD14_CACHE[model_key]

    # Evict the previous session before allocating a new one.
    if _WD14_CACHE:
        prev_key = next(iter(_WD14_CACHE))
        prev_sess, _prev_rows = _WD14_CACHE.pop(prev_key)
        try:
            # onnxruntime InferenceSession has no public close(); drop the
            # underlying C++ handle so the CUDA allocation is released.
            if hasattr(prev_sess, "_sess"):
                prev_sess._sess = None
        except Exception:
            pass
        del prev_sess
        import gc
        gc.collect()
        if progress_cb:
            progress_cb("Released previous WD14 model ({})".format(prev_key.split(" ", 1)[0]))

    model_p, tags_p = _wd14_download(model_key, progress_cb)

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime not installed. Restart Forge once after the "
            "extension installs - install.py pip-installs onnxruntime "
            "automatically."
        ) from e

    if progress_cb: progress_cb("Loading {} into memory...".format(model_key))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_p, providers=providers)

    import csv
    rows = []
    with open(tags_p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "name":     row["name"],
                "category": int(row["category"]),
                "count":    int(row.get("count", 0) or 0),
            })

    _WD14_CACHE[model_key] = (sess, rows)
    return sess, rows


def _wd14_preprocess(pil_img, target=WD14_TARGET_SIZE):
    """Resize-with-pad to a square, convert to BGR float32, batch-dim."""
    from PIL import Image
    import numpy as np
    img = pil_img.convert("RGB")
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    canvas = canvas.resize((target, target), Image.BICUBIC)
    arr = np.array(canvas, dtype=np.float32)
    arr = arr[:, :, ::-1]                     # RGB -> BGR (SmilingWolf convention)
    arr = np.expand_dims(arr, axis=0)         # add batch dim
    return arr


def _parse_character_tag(tag):
    """Parse a Danbooru character tag into (name, series_or_None).
    Examples:
      kafka_(honkai:_star_rail) -> ('kafka', 'honkai:_star_rail')
      seele_vollerei            -> ('seele_vollerei', None)
      hatsune_miku              -> ('hatsune_miku', None)
      jeanne_d'arc_(fate)       -> ("jeanne_d'arc", 'fate')
    """
    import re
    m = re.match(r"^(.+?)_\(([^)]+)\)$", tag)
    if m:
        return m.group(1), m.group(2)
    return tag, None


def _humanize_name(s):
    """Convert a booru underscored name to human-readable Title Case.
    Preserves embedded colons (e.g. 'honkai:_star_rail' -> 'Honkai: Star Rail').
    """
    if not s:
        return s
    out = s.replace("_", " ").strip()
    # Title-case each word; preserve apostrophes and existing punctuation
    words = out.split(" ")
    cap = []
    for w in words:
        if not w:
            continue
        # Handle words like "d'arc" (keep d lowercase, Arc cased)
        if "'" in w and len(w) > 1 and w[1] == "'":
            cap.append(w[0].lower() + "'" + w[2:].capitalize())
        else:
            cap.append(w[0].upper() + w[1:] if w else "")
    return " ".join(cap)


def _format_character_display(tag, score):
    """Render a character tag for display: 'Name - Series (NN%)'."""
    name, series = _parse_character_tag(tag)
    pretty_name = _humanize_name(name)
    pretty_series = _humanize_name(series) if series else None
    if pretty_series:
        return "{} - {} ({:.0%})".format(pretty_name, pretty_series, score)
    return "{} ({:.0%})".format(pretty_name, score)


def _scan_metadata_for_characters(metadata_positive, all_char_predictions):
    """If the metadata's positive prompt mentions a character that WD14 also
    saw (at ANY confidence, not just above the threshold), promote those.
    Returns list of (tag, score, source='metadata-confirmed').
    """
    if not metadata_positive:
        return []
    pos_lower = metadata_positive.lower()
    promoted = []
    for tag, score in all_char_predictions:
        # Try exact tag match and name-only match against the prompt
        name, _ = _parse_character_tag(tag)
        if tag.lower() in pos_lower or name.lower() in pos_lower:
            promoted.append((tag, max(score, 0.99)))  # promote to high confidence
    return promoted


def analyze_image_wd14(pil_img, general_threshold=0.35, character_threshold=0.55,
                       artist_threshold=0.30, model_key=None, progress_cb=None):
    """Return {'general':..., 'characters':..., 'artists':..., 'styles':...,
              'ratings':..., 'tag_string':str}.
    Each list is [(tag, score)] sorted by score descending.
    'styles' is a subset of 'general' filtered against WD14_STYLE_TAGS so the
    user can see "this looks like watercolor / cel-shading / etc." at a glance.
    """
    if not model_key:
        model_key = WD14_DEFAULT_MODEL_KEY
    sess, tags = _wd14_load(model_key, progress_cb)
    arr = _wd14_preprocess(pil_img)
    if progress_cb: progress_cb("Running inference...")

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    preds = sess.run([output_name], {input_name: arr})[0][0]

    ratings    = []
    characters = []
    characters_all = []  # ALL character predictions, no threshold, for x-ref
    artists    = []
    general    = []
    for i, score in enumerate(preds):
        tag = tags[i]
        cat = tag["category"]
        s   = float(score)
        nm  = tag["name"]
        if   cat == 9: ratings.append((nm, s))
        elif cat == 4:
            characters_all.append((nm, s))   # always collect, regardless of threshold
            if s >= character_threshold:
                characters.append((nm, s))
        elif cat == 1:               # artist
            if s >= artist_threshold:
                artists.append((nm, s))
        else:                         # 0 = general, also species/copyright
            if s >= general_threshold:
                general.append((nm, s))

    general.sort(key=lambda t: -t[1])
    characters.sort(key=lambda t: -t[1])
    characters_all.sort(key=lambda t: -t[1])
    artists.sort(key=lambda t: -t[1])
    ratings.sort(key=lambda t: -t[1])

    # Pull out style-descriptor tags from the general bucket.
    style_lower = {s.lower() for s in WD14_STYLE_TAGS}
    styles = [(nm, sc) for (nm, sc) in general if nm.lower() in style_lower]
    # general_clean = general minus style tags (so they don't double-count in
    # the main tag string).
    general_clean = [(nm, sc) for (nm, sc) in general if nm.lower() not in style_lower]

    # Replication string order (matches Illustrious caption convention):
    #   characters -> artists -> general (minus style) -> style descriptors
    parts = ([t[0] for t in characters]
             + [t[0] for t in artists]
             + [t[0] for t in general_clean]
             + [t[0] for t in styles])
    tag_string = ", ".join(parts)

    # Scene context: group sub-buckets pulled from general_clean using the
    # category whitelists. Lower the effective threshold here so weak signals
    # (e.g. "golden_hour" at 0.28) still show up — these tags matter even at
    # low confidence.
    scene_buckets = {
        "Time":     SCENE_TIME_TAGS,
        "Weather":  SCENE_WEATHER_TAGS,
        "Lighting": SCENE_LIGHTING_TAGS,
        "Location": SCENE_LOCATION_TAGS,
        "Season":   SCENE_SEASON_TAGS,
        "Sky":      SCENE_SKY_TAGS,
    }
    # For scene tags we also peek into the raw predictions at a lower
    # threshold (0.20) so we don't miss weak-but-correct environment cues.
    SCENE_LOW_THRESHOLD = 0.20
    scene_low_general = []
    for i, score in enumerate(preds):
        tag = tags[i]
        if tag["category"] != 0:
            continue
        s = float(score)
        if s >= SCENE_LOW_THRESHOLD:
            scene_low_general.append((tag["name"], s))

    scene_context = {}
    for label, whitelist in scene_buckets.items():
        wl_lower = {t.lower() for t in whitelist}
        hits = [(nm, sc) for (nm, sc) in scene_low_general
                if nm.lower() in wl_lower]
        if hits:
            hits.sort(key=lambda t: -t[1])
            scene_context[label] = hits

    return {
        "general":         general_clean,
        "characters":      characters,
        "characters_all":  characters_all,    # full prediction list for x-ref
        "artists":         artists,
        "styles":          styles,
        "ratings":         ratings,
        "scene_context":   scene_context,
        "tag_string":      tag_string,
    }


def _parse_a1111_params(text):
    """Parse an A1111 / Forge-style parameters block.
    Returns {'positive', 'negative', 'settings': {k: v}}.
    Tolerant of missing fields."""
    import re
    out = {"positive": "", "negative": "", "settings": {}}
    if not text or not text.strip():
        return out
    text = text.strip()

    # Split positive from negative-and-rest
    if "Negative prompt:" in text:
        pos_part, rest = text.split("Negative prompt:", 1)
        out["positive"] = pos_part.strip()
        # Find where the settings line starts. Settings keys always come at the
        # start of a new line and look like "Steps: 28, Sampler: Euler a, ...".
        m = re.search(r'\n(Steps|Sampler|CFG scale|Seed|Size|Model|VAE|'
                       r'Schedule type|Denoising strength|Clip skip|Hires|'
                       r'ControlNet)[ a-zA-Z]*: ', rest)
        if m:
            out["negative"] = rest[:m.start()].strip()
            settings_text  = rest[m.start():].lstrip()
        else:
            out["negative"] = rest.strip()
            settings_text  = ""
    else:
        m = re.search(r'\n(Steps|Sampler|CFG scale|Seed|Size|Model|VAE|'
                       r'Schedule type)[ a-zA-Z]*: ', text)
        if m:
            out["positive"] = text[:m.start()].strip()
            settings_text  = text[m.start():].lstrip()
        else:
            out["positive"] = text.strip()
            settings_text  = ""

    # Parse settings. Values can contain commas inside quotes; A1111 doesn't
    # quote consistently. We split on ", " followed by Capitalized key + ":".
    if settings_text:
        pairs = re.split(r',\s+(?=[A-Z][\w\s\-]*?:\s)', settings_text)
        for pair in pairs:
            if ":" not in pair:
                continue
            k, v = pair.split(":", 1)
            out["settings"][k.strip()] = v.strip().rstrip(",")
    return out


def _parse_comfyui_workflow(workflow):
    """Walk a ComfyUI workflow/prompt JSON. Returns positive/negative/settings.
    Picks the longest CLIPTextEncode text as positive; nodes whose title hints
    'negative' are treated as negative."""
    out = {"positive": "", "negative": "", "settings": {}}
    if not isinstance(workflow, dict):
        return out
    pos_candidates, neg_candidates = [], []
    for _, node in workflow.items():
        if not isinstance(node, dict): continue
        class_type = node.get("class_type", "") or ""
        inputs     = node.get("inputs", {}) or {}
        meta       = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
        title      = (meta.get("title", "") or "").lower()

        if "CLIPTextEncode" in class_type:
            t = inputs.get("text", "")
            if isinstance(t, str) and t.strip():
                if "negative" in title or "neg" in title:
                    neg_candidates.append(t)
                else:
                    pos_candidates.append(t)

        if "KSampler" in class_type or class_type.endswith("Sampler"):
            for k in ("steps", "cfg", "sampler_name", "scheduler",
                       "seed", "denoise"):
                if k in inputs and not isinstance(inputs[k], list):
                    out["settings"][k] = inputs[k]

    if pos_candidates:
        out["positive"] = max(pos_candidates, key=len).strip()
    if neg_candidates:
        out["negative"] = max(neg_candidates, key=len).strip()
    return out


def _parse_novelai_metadata(text_chunks):
    """Parse NovelAI PNG metadata. Returns {'positive', 'negative', 'settings'}.

    NAI's metadata lives in:
      - Description: positive prompt (sometimes truncated)
      - Comment: full JSON with prompt, uc, sampler, scale, seed, w/h, etc.
      - v4 PNGs additionally have v4_prompt / v4_negative_prompt structured fields
        with per-character captions inside data["v4_prompt"]["caption"].
    """
    out = {"positive": "", "negative": "", "settings": {}}
    if not text_chunks:
        return out

    # ---- Description = positive prompt (NAI v3 + v4 fallback) ----
    desc = text_chunks.get("Description", "")
    if desc and isinstance(desc, str):
        out["positive"] = desc.strip()

    # ---- Comment = full generation JSON ----
    comment = text_chunks.get("Comment", "")
    if not comment or not isinstance(comment, str):
        # Fall back to whatever Description gave us
        return out
    try:
        import json
        data = json.loads(comment)
    except Exception:
        return out
    if not isinstance(data, dict):
        return out

    # Prefer the Comment JSON's "prompt" field if present (more reliable than
    # the Description field, which can be truncated).
    if isinstance(data.get("prompt"), str) and data["prompt"].strip():
        out["positive"] = data["prompt"].strip()

    # NAI v4 structured prompt overrides scalar "prompt".
    v4p = data.get("v4_prompt")
    if isinstance(v4p, dict):
        cap = v4p.get("caption") if isinstance(v4p.get("caption"), dict) else None
        if cap:
            base = cap.get("base_caption", "")
            if isinstance(base, str) and base.strip():
                out["positive"] = base.strip()
            # Layer per-character captions
            chars = cap.get("char_captions", [])
            if isinstance(chars, list) and chars:
                char_strings = []
                for c in chars:
                    if isinstance(c, dict):
                        cc = c.get("char_caption", "")
                        if isinstance(cc, str) and cc.strip():
                            char_strings.append(cc.strip())
                if char_strings:
                    out["positive"] += " | " + " | ".join(char_strings)

    # Negative prompt: "uc" (NAI v3), then "negative_prompt", then v4_negative_prompt
    if isinstance(data.get("uc"), str) and data["uc"].strip():
        out["negative"] = data["uc"].strip()
    elif isinstance(data.get("negative_prompt"), str) and data["negative_prompt"].strip():
        out["negative"] = data["negative_prompt"].strip()

    v4np = data.get("v4_negative_prompt")
    if isinstance(v4np, dict):
        cap = v4np.get("caption") if isinstance(v4np.get("caption"), dict) else None
        if cap:
            base = cap.get("base_caption", "")
            if isinstance(base, str) and base.strip():
                out["negative"] = base.strip()

    # ---- Settings mapping (NAI -> A1111-like keys for consistent display) ----
    settings_map = [
        ("steps",                 "Steps"),
        ("scale",                 "CFG scale"),
        ("sampler",               "Sampler"),
        ("seed",                  "Seed"),
        ("noise_schedule",        "Schedule"),
        ("cfg_rescale",           "CFG rescale"),
        ("uncond_scale",          "Uncond scale"),
        ("guidance",              "Guidance"),
        ("guidance_rescale",      "Guidance rescale"),
        ("dynamic_thresholding",  "Dynamic thresholding"),
        ("dynamic_thresholding_percentile", "DT percentile"),
        ("dynamic_thresholding_mimic_scale", "DT mimic scale"),
        ("strength",              "Strength"),
        ("noise",                 "Noise"),
        ("model",                 "Model"),
    ]
    for nai_k, our_k in settings_map:
        if nai_k in data and data[nai_k] not in (None, ""):
            out["settings"][our_k] = data[nai_k]

    # Combine width/height into Size
    w = data.get("width"); h = data.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        out["settings"]["Size"] = "{}x{}".format(int(w), int(h))

    # Include the SDXL base / NAI model hint from Source if present
    src_field = text_chunks.get("Source", "")
    if src_field and "Model" not in out["settings"]:
        out["settings"]["Source"] = src_field

    return out


def _extract_image_metadata(pil_img):
    """Stage 1 of the ultimate analyzer.
    Returns {'source': 'a1111|forge|comfyui|none|exif',
             'positive': str, 'negative': str, 'settings': dict,
             'exif': dict, 'raw_text_chunks': dict}."""
    out = {"source": "none", "positive": "", "negative": "",
           "settings": {}, "exif": {}, "raw_text_chunks": {}}

    if pil_img is None:
        return out

    # PNG text chunks (Pillow stores them in img.text after Image.open).
    text_chunks = {}
    try:
        if hasattr(pil_img, "text") and isinstance(pil_img.text, dict):
            text_chunks = dict(pil_img.text)
        if hasattr(pil_img, "info") and isinstance(pil_img.info, dict):
            # PIL also dumps PNG tEXt/zTXt chunks into .info
            for k, v in pil_img.info.items():
                if k not in text_chunks and isinstance(v, str):
                    text_chunks[k] = v
    except Exception:
        pass
    out["raw_text_chunks"] = text_chunks

    # A1111 / Forge format takes precedence over NovelAI signature detection.
    # Rationale: if an image was re-saved in A1111/Forge after NAI generation,
    # the "Software"/"Source" tags may still say NovelAI but the authoritative
    # prompt + negative now live in the "parameters" chunk. Trusting NAI when
    # an A1111 parameters block exists drops the negative prompt silently.
    if "parameters" in text_chunks and isinstance(text_chunks["parameters"], str):
        parsed = _parse_a1111_params(text_chunks["parameters"])
        if parsed["positive"]:
            out["source"] = "a1111/forge"
            out["positive"] = parsed["positive"]
            out["negative"] = parsed["negative"]
            out["settings"] = parsed["settings"]

    # NovelAI format — only fall through to this if no A1111 parameters block
    # was found AND the NAI Comment chunk (which holds the real settings JSON)
    # actually parses. The Software/Source-only check was too loose.
    if out["source"] == "none" and "Comment" in text_chunks:
        try:
            import json as _json
            _json.loads(text_chunks["Comment"])  # validate
            parsed = _parse_novelai_metadata(text_chunks)
            if parsed["positive"]:
                out["source"] = "novelai"
                out["positive"] = parsed["positive"]
                out["negative"] = parsed["negative"]
                out["settings"] = parsed["settings"]
        except (ValueError, TypeError):
            pass

    # A1111 / Forge format (re-check kept for backward compat with the original
    # control flow — harmless no-op if we already populated above).
    if out["source"] == "none" and "parameters" in text_chunks and isinstance(text_chunks["parameters"], str):
        parsed = _parse_a1111_params(text_chunks["parameters"])
        if parsed["positive"]:
            out["source"] = "a1111/forge"
            out["positive"] = parsed["positive"]
            out["negative"] = parsed["negative"]
            out["settings"] = parsed["settings"]

    # ComfyUI format
    if out["source"] == "none":
        for k in ("workflow", "prompt"):
            if k in text_chunks and isinstance(text_chunks[k], str):
                try:
                    import json
                    wf = json.loads(text_chunks[k])
                    parsed = _parse_comfyui_workflow(wf)
                    if parsed["positive"]:
                        out["source"] = "comfyui"
                        out["positive"] = parsed["positive"]
                        out["negative"] = parsed["negative"]
                        out["settings"] = parsed["settings"]
                        break
                except Exception:
                    continue

    # EXIF (always extract; useful for photos)
    try:
        exif = pil_img.getexif() if hasattr(pil_img, "getexif") else {}
        if exif:
            from PIL.ExifTags import TAGS
            for tag_id, val in exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                out["exif"][tag] = val
            if out["source"] == "none" and out["exif"]:
                out["source"] = "exif"
    except Exception:
        pass

    return out


def build_ultimate_prompt(metadata, wd14_result, artist_weight=1.20,
                           style_weight=1.15, add_quality=True):
    """Stage 3: merge metadata + WD14 into the most complete prompt.

    Returns (positive_str, negative_str, info_dict).

    Strategy:
      - If metadata has a positive prompt -> that's the authoritative base.
        WD14 only ADDS what's missing.
      - If no metadata -> WD14 is everything.
      - Artist + style tags get weight emphasis.
      - Deduplication is case-insensitive against an accumulating lowercase set.
    """
    info = {"used_metadata": False, "layers": []}

    seen_lower = set()
    parts = []

    def _split(s):
        return [t.strip() for t in s.split(",") if t.strip()]

    def _push(items, layer_label=None, wrap_strength=None):
        added = 0
        for it in items:
            low = it.lower().lstrip("(").rstrip(")").split(":")[0].strip()
            if not low or low in seen_lower:
                continue
            seen_lower.add(low)
            if wrap_strength is not None and abs(wrap_strength - 1.0) > 0.01:
                parts.append("({}:{:.2f})".format(it, wrap_strength))
            else:
                parts.append(it)
            added += 1
        if layer_label and added:
            info["layers"].append("{}: +{}".format(layer_label, added))

    # ---- Layer A: original metadata positive (authoritative base) ----
    meta_pos = (metadata or {}).get("positive", "").strip()
    if meta_pos:
        _push(_split(meta_pos), layer_label="metadata original")
        info["used_metadata"] = True

    # ---- Layer B: WD14 artist tags (weighted) ----
    artists = (wd14_result or {}).get("artists", []) or []
    if artists:
        _push([a[0] for a in artists[:5]],
              layer_label="WD14 artist",
              wrap_strength=artist_weight)

    # ---- Layer C: WD14 characters ----
    chars = (wd14_result or {}).get("characters", []) or []
    if chars:
        _push([c[0] for c in chars], layer_label="WD14 character")

    # ---- Layer D: scene context (time/location/lighting/weather/season/sky) ----
    sc = (wd14_result or {}).get("scene_context", {}) or {}
    scene_picks = []
    for label in ("Time", "Location", "Lighting", "Weather", "Season", "Sky"):
        for tag, _ in (sc.get(label) or [])[:3]:
            scene_picks.append(tag)
    if scene_picks:
        _push(scene_picks, layer_label="WD14 scene")

    # ---- Layer E: WD14 style descriptors (weighted) ----
    styles = (wd14_result or {}).get("styles", []) or []
    if styles:
        _push([s[0] for s in styles[:8]],
              layer_label="WD14 style",
              wrap_strength=style_weight)

    # ---- Layer F: WD14 general tags ----
    general = (wd14_result or {}).get("general", []) or []
    if general:
        _push([g[0] for g in general[:20]], layer_label="WD14 general")

    # ---- Layer G: quality boilerplate ----
    if add_quality:
        _push(_split("masterpiece, best quality, very aesthetic, absurdres, newest"),
              layer_label="quality")

    positive = ", ".join(parts)

    # ---- Negative ----
    meta_neg = (metadata or {}).get("negative", "").strip()
    if meta_neg:
        negative = meta_neg
        info["negative_source"] = "metadata"
    else:
        negative = ("worst quality, low quality, normal quality, lowres, "
                    "bad anatomy, bad hands, watermark, signature, "
                    "jpeg artifacts, blurry, sketch")
        info["negative_source"] = "default"

    return positive, negative, info


def _artist_emit(label: str, strength: float = 1.0) -> str:
    """Return the artist tag formatted for the prompt.
    Strength of 1.0 emits the bare tag; other strengths wrap as (tag:strength)."""
    entry = ARTIST_STYLES.get(label)
    if not entry: return ""
    tag = entry[0]
    if not tag: return ""
    if abs(strength - 1.0) < 0.01:
        return tag
    return "({}:{:.2f})".format(tag, strength)


def _artist_preview_md(label: str) -> str:
    """Build the markdown preview shown under the dropdown."""
    entry = ARTIST_STYLES.get(label)
    if not entry:
        return "*Pick an artist to see the description.*"
    tag, desc, url = entry
    if not tag:
        return "*{}*".format(desc)
    md = "**Tag**: `{}`  \n**Style**: {}".format(tag, desc)
    if url:
        md += "  \n[View sample images on Danbooru]({})".format(url)
    return md


# ===========================================================================
# 9. Body / Physique
# ===========================================================================
# Mix of safe (slim, fit, athletic, hourglass...) and [NSFW] bust / hip
# specifics. Picked separately so the user can mix any body type with any
# outfit/pose/scene.

BODY_TYPES = {
    "Slim / petite":           "slim, petite, slender",
    "Average / fit":           "fit, average_build",
    "Athletic / toned":        "athletic, toned, fit",
    "Curvy":                   "curvy, voluptuous",
    "Hourglass":               "hourglass_figure, narrow_waist, wide_hips",
    "Muscular / abs":          "muscular, abs, toned_stomach",
    "Plump / thicc":           "thick_thighs, wide_hips, plump",
    "Tall":                    "tall",
    "Short":                   "short, petite",
    "[NSFW] Small bust":       "small_breasts",
    "[NSFW] Medium bust":      "medium_breasts",
    "[NSFW] Large bust":       "large_breasts",
    "[NSFW] Huge bust":        "huge_breasts",
    "[NSFW] Gigantic bust":    "gigantic_breasts",
    "[NSFW] Flat chest":       "flat_chest",
    "[NSFW] Big ass":          "huge_ass, wide_hips",
    "[NSFW] Thick thighs":     "thick_thighs",
    "Lean / wiry":             "lean, wiry",
    "Soft / round":            "soft_body, round",
    "Tomboyish (boyish)":      "tomboy, boyish_figure",
    "Statuesque (very tall)":  "tall, statuesque, long_legs",
    "Athletic with curves":    "fit, hourglass_figure, athletic",
    "Pixie / fairy small":     "small, petite, tiny_frame",
    "Pear shaped":             "wide_hips, narrow_shoulders",
    "Apple shaped":            "wide_shoulders, broad_chest",
    NONE:                      "",
}

# ===========================================================================
# 10. Intimacy / NSFW level
# ===========================================================================
# Tier of explicitness. Adds rating + content tags that nudge the model into
# the right zone. Stays at "(safe — none)" by default.

INTIMACY_LEVELS = {
    "(safe - none)":              "",
    "Romantic / sensual":          "sensual, intimate, romantic",
    "Suggestive (mildly sexy)":    "suggestive, sexy",
    "[NSFW] Softcore":             "nsfw, sensual, intimate, sexy, partially_nude",
    "[NSFW] Erotic":               "nsfw, erotic, seductive, naked, undressed",
    "[NSFW] Explicit (sex acts)":  "nsfw, explicit, sex, intercourse",
    "[NSFW] Hardcore":             "nsfw, explicit, hardcore, sex, intercourse",
    "[NSFW] Fetish - bondage":     "nsfw, bondage, restrained, rope",
    "[NSFW] Fetish - femdom":      "nsfw, femdom, dominant_female",
    "[NSFW] Aftermath":            "nsfw, after_sex, cum, sweat, disheveled",
}


# ===========================================================================
# Quality / Negative
# ===========================================================================

ILLUSTRIOUS_QUALITY_TIERS = {
    "Best (most opinionated)":      "masterpiece, best quality, amazing quality, very aesthetic, absurdres",
    "Standard (safe default)":      "masterpiece, best quality, very aesthetic, absurdres",
    "Minimal (just say it's good)": "best quality, very aesthetic",
    "None (subject only)":          "",
}

NOOBAI_QUALITY_TIERS = {
    "Best (most opinionated)":      "masterpiece, best quality, very aesthetic, absurdres, newest",
    "Standard (safe default)":      "masterpiece, best quality, very aesthetic",
    "Minimal (just say it's good)": "best quality",
    "None (subject only)":          "",
}

ILLUSTRIOUS_NEGATIVE_TIERS = {
    "Strong (thorough)": (
        "worst quality, low quality, normal quality, lowres, "
        "bad anatomy, bad hands, bad fingers, extra fingers, missing fingers, "
        "extra arms, extra legs, missing arms, missing legs, "
        "mutated hands, fused fingers, "
        "watermark, signature, artist name, text, error, "
        "jpeg artifacts, blurry, sketch, "
        "censored, mosaic censoring, bar censor"
    ),
    "Standard (safe default)": (
        "worst quality, low quality, normal quality, lowres, "
        "bad anatomy, bad hands, watermark, signature, "
        "jpeg artifacts, blurry, sketch"
    ),
    "Minimal (just block junk)": "worst quality, low quality, lowres, jpeg artifacts",
    "None (no negative)": "",
}

NOOBAI_NEGATIVE_TIERS = ILLUSTRIOUS_NEGATIVE_TIERS

YEAR_TAGS = ["(no year tag)", "newest", "recent", "mid", "old", "oldest"]
SOURCE_TAGS = [
    "(no source tag)", "source_anime", "source_cartoon",
    "source_pony", "source_furry", "source_3d", "source_real",
    "official_art",
]
RATING_TAGS = [
    "(no rating tag)", "rating:general", "rating:sensitive",
    "rating:questionable", "rating:explicit",
]

MODELS: Dict[str, Dict] = {
    "Illustrious-XL (any variant)": {
        "keywords": ["illustrious", "illustrius", "waiillustrious", "wai_illustrious"],
        "positive_tiers": ILLUSTRIOUS_QUALITY_TIERS,
        "negative_tiers": ILLUSTRIOUS_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_ILLUSTRIOUS,
    },
    "NoobAI XL (v-pred / eps)": {
        "keywords": ["noobai", "noob_ai", "noob-ai"],
        "positive_tiers": NOOBAI_QUALITY_TIERS,
        "negative_tiers": NOOBAI_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_NOOBAI,
    },
    "(detect from loaded checkpoint)": {
        "keywords": [],
        "positive_tiers": ILLUSTRIOUS_QUALITY_TIERS,
        "negative_tiers": ILLUSTRIOUS_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_ILLUSTRIOUS,
    },
}

# ===========================================================================
# Scenario templates
# ===========================================================================

SCENARIOS: Dict[str, Dict] = {
    "(none — design from scratch)": {},

    "Character portrait (clean)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Portrait (head/shoulders)",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Detailed anime (default)",
    },
    "Full-body reference sheet": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, arms crossed",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Neutral",
        "style":   "Detailed anime (default)",
    },
    "School day (classroom)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (sailor)",
        "location":"Classroom",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Smiling",
        "style":   "Detailed anime (default)",
    },
    "Cafe outing (casual)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Beach vacation": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Swimsuit (bikini)",
        "location":"Beach",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "Big grin",
        "style":   "Vibrant anime",
    },
    "Combat / action scene": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Combat / fighting stance",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Dutch angle",
        "outfit":  "Fantasy armor",
        "location":"Mountain landscape",
        "time":    "Sunset",
        "lighting":"Cinematic",
        "expr":    "Determined",
        "style":   "Cinematic / film",
    },
    "Magical girl transformation": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Spell-casting",
        "action":  "Looking up",
        "shot":    "Full body",
        "angle":   "From below",
        "outfit":  "Magical girl",
        "location":"Starry sky",
        "time":    "Late night",
        "lighting":"Volumetric / god rays",
        "expr":    "Confident",
        "style":   "Vibrant anime",
    },
    "Idol performance": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Dancing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Idol stage costume",
        "location":"Stage / concert",
        "time":    NONE,
        "lighting":"Neon glow",
        "expr":    "Big grin",
        "style":   "Vibrant anime",
    },
    "Summer festival (yukata)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Walking",
        "action":  "Looking back",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Yukata (summer festival)",
        "location":"Festival night",
        "time":    "Night",
        "lighting":"Candle / firelight",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Cozy bedroom (pajamas)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Pajamas",
        "location":"Bedroom",
        "time":    "Late night",
        "lighting":"Candle / firelight",
        "expr":    "Sleepy",
        "style":   "Soft pastel anime",
    },
    "Office worker": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Holding phone",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Office / business",
        "location":"Office",
        "time":    "Morning",
        "lighting":"Bright / sunny",
        "expr":    "Confident",
        "style":   "Detailed anime (default)",
    },
    "Cyberpunk night": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands in pockets",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Cyberpunk",
        "location":"Cyberpunk city",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Smug",
        "style":   "Cinematic / film",
    },
    "Sci-fi space": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Sci-fi bodysuit",
        "location":"Space station",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Confident",
        "style":   "3D / CG render",
    },
    "Fantasy adventurer": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Fantasy armor",
        "location":"Forest",
        "time":    "Golden hour",
        "lighting":"Volumetric / god rays",
        "expr":    "Determined",
        "style":   "Painterly fantasy",
    },
    "Sakura / cherry blossoms": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (blazer)",
        "location":"Cherry blossom",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Rainy day (umbrella)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Holding something",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Rainy street",
        "time":    "Twilight",
        "lighting":"Moody / low light",
        "expr":    "Soft smile",
        "style":   "Cinematic / film",
    },
    "Winter snow": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Winter coat",
        "location":"Snowy outdoors",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Goth / dark portrait": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Goth / lolita",
        "location":"Simple black background",
        "time":    NONE,
        "lighting":"Moody / low light",
        "expr":    "Serious",
        "style":   "Gothic / dark",
    },
    # Multi-character scenarios (test that 1girl doesn't show up)
    "Two girls hanging out": {
        "subject": "Pair - 2girls",
        "pose":    "Sitting on chair",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Big grin",
        "style":   "Detailed anime (default)",
    },
    "Mixed pair (1girl + 1boy)": {
        "subject": "Pair - 1girl + 1boy",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Park (outdoor)",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Detailed anime (default)",
    },
    # --- [NSFW] scenario presets ---
    "[NSFW] Bedroom seduction": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "[NSFW] Reclining on bed",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Lingerie (lacy + garter)",
        "location":"Bedroom (bed close-up)",
        "time":    "Late night",
        "lighting":"Candle / firelight",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Cinematic / film",
    },
    "[NSFW] Shower scene": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Naked (no clothes)",
        "location":"Shower",
        "time":    NONE,
        "lighting":"Soft natural",
        "expr":    "[NSFW] Erotic blush",
        "style":   "Cinematic / film",
    },
    "[NSFW] Hot spring relax": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on knees (seiza)",
        "action":  "Closed eyes",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Naked (no clothes)",
        "location":"Hot spring / onsen",
        "time":    "Twilight",
        "lighting":"Volumetric / god rays",
        "expr":    "Soft smile",
        "style":   "Painterly fantasy",
    },
    "[NSFW] Lingerie photoshoot": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Lingerie (lacy + garter)",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Smug",
        "style":   "Cinematic / film",
    },
    "[NSFW] Naked apron (kitchen)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking back",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From behind",
        "outfit":  "[NSFW] Naked apron",
        "location":"Kitchen",
        "time":    "Morning",
        "lighting":"Soft natural",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Detailed anime (default)",
    },
    "[NSFW] Bunny suit casino": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Wink",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Playboy bunny suit",
        "location":"Bar / nightclub",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Smug",
        "style":   "Vibrant anime",
    },
    "[NSFW] Beach micro bikini": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands behind back",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Micro bikini",
        "location":"Beach",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Vibrant anime",
    },
    "Library study session": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting at desk",
        "action":  "Studying",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Dark academia",
        "location":"Library, antique",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Concentrating",
        "style":   "Cinematic / film",
    },
    "Park picnic": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Sundress + sun hat",
        "location":"Park (outdoor)",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Movie night couch": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Pajamas",
        "location":"Living room",
        "time":    "Late night",
        "lighting":"Spotlight (single)",
        "expr":    "Content / peaceful",
        "style":   "Cinematic / film",
    },
    "Karaoke night": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Singing",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Karaoke room",
        "time":    "Night",
        "lighting":"Neon glow",
        "expr":    "Excited",
        "style":   "Vibrant anime",
    },
    "Sushi date": {
        "subject": "Pair - 1girl + 1boy",
        "pose":    "Sitting on chair",
        "action":  "Eating",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Restaurant",
        "time":    "Twilight",
        "lighting":"Lantern light",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Rainy day cafe": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Trench coat",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Moody / low light",
        "expr":    "Daydreaming",
        "style":   "Cinematic / film",
    },
    "Sunset rooftop": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking away",
        "shot":    "Full body",
        "angle":   "From side",
        "outfit":  "Casual streetwear",
        "location":"Rooftop",
        "time":    "Sunset",
        "lighting":"Sunset glow",
        "expr":    "Daydreaming",
        "style":   "Cinematic / film",
    },
    "Camping fireside": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting cross-legged",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Forest",
        "time":    "Night",
        "lighting":"Candle / firelight",
        "expr":    "Content / peaceful",
        "style":   "Painterly fantasy",
    },
    "Concert front row": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Cheering",
        "shot":    "Upper body",
        "angle":   "From below",
        "outfit":  "Casual streetwear",
        "location":"Stage / concert",
        "time":    NONE,
        "lighting":"Neon glow",
        "expr":    "Excited",
        "style":   "Cinematic / film",
    },
    "New Year fireworks": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking up",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Yukata (summer festival)",
        "location":"Festival night",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Eyes wide with awe",
        "style":   "Vibrant anime",
    },
    "Group scene (multiple girls)": {
        "subject": "Group - multiple girls",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (blazer)",
        "location":"School courtyard",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Smiling",
        "style":   "Detailed anime (default)",
    },
}

# ===========================================================================
# Detection
# ===========================================================================

def _get_loaded_checkpoint() -> str:
    try:
        if shared is not None:
            ckpt = getattr(shared.opts, "sd_model_checkpoint", "") or ""
            return str(ckpt)
    except Exception:
        pass
    return ""


def _detect_model_family(checkpoint_name: str) -> str:
    name_low = (checkpoint_name or "").lower()
    if not name_low:
        return "(detect from loaded checkpoint)"
    for family_label, family in MODELS.items():
        for kw in family["keywords"]:
            if kw in name_low:
                return family_label
    return "Illustrious-XL (any variant)"


def _resolve_family(family_label: str):
    if family_label == "(detect from loaded checkpoint)" or family_label not in MODELS:
        family_label = _detect_model_family(_get_loaded_checkpoint())
    if family_label not in MODELS or family_label == "(detect from loaded checkpoint)":
        family_label = "Illustrious-XL (any variant)"
    return family_label, MODELS[family_label]


def _count_from_subject(label: str) -> int:
    """How many per-character trait slots are relevant.
    Solo / (skip) = 1, Pair = 2, Trio = 3, Group = up to 4."""
    if label.startswith("Group"): return 4
    if label.startswith("Trio"):  return 3
    if label.startswith("Pair"):  return 2
    return 1


# Subjects that are all-female (no males implied). The female-only negative
# helper kicks in when subject is in this set and the user has the checkbox on.
ALL_FEMALE_SUBJECTS = {
    "Solo - 1girl (young adult female)",
    "Solo - mature woman",
    "Pair - 2girls",
    "Trio - 3girls",
    "Group - multiple girls",
}

# Subjects with two or more females (eligible for the yuri tag).
MULTI_FEMALE_SUBJECTS = {
    "Pair - 2girls",
    "Trio - 3girls",
    "Group - multiple girls",
}

# Tag string we add to the NEGATIVE prompt when 'Force female-only' is on.
# Comprehensive male-exclusion so Illustrious can't slip a boy in.
FEMALE_ONLY_NEGATIVE = (
    "1boy, 2boys, 3boys, multiple_boys, male, male_focus, penis, testicles, "
    "mustache, beard, facial_hair"
)

# Tag string we add to the POSITIVE prompt when 'Yuri' is on.
YURI_POSITIVE = "yuri"


def _is_all_female(subject_label):
    return subject_label in ALL_FEMALE_SUBJECTS


def _is_multi_female(subject_label):
    return subject_label in MULTI_FEMALE_SUBJECTS


def _split_names(s: str):
    """Comma-split that ignores commas inside parens.
    Needed so series names like 'kafka_(honkai:_star_rail)' survive intact."""
    parts, depth, buf = [], 0, []
    for c in s:
        if c == '(':
            depth += 1; buf.append(c)
        elif c == ')':
            depth = max(0, depth - 1); buf.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(buf)); buf = []
        else:
            buf.append(c)
    if buf: parts.append(''.join(buf))
    return parts


def _parse_char_names(s: str):
    """Parse one or more character names into normalized booru-format strings.
    Returns a list (possibly empty). Spaces -> underscores. Keeps parens, colons,
    hyphens, underscores."""
    if not s: return []
    out = []
    for part in _split_names(s):
        n = part.strip().lower().replace(" ", "_")
        n = re.sub(r"[^\w\(\):_\-]", "", n)
        if n: out.append(n)
    return out


# ===========================================================================
# Assembly
# ===========================================================================

def _tag(d: Dict[str, str], label: str) -> str:
    if not label or label == NONE:
        return ""
    return d.get(label, "")


def _join(*parts: str) -> str:
    """Comma-join parts, splitting each on commas, deduping case-insensitively,
    preserving first-seen order."""
    seen = []
    seen_set = set()
    for p in parts:
        if not p:
            continue
        for t in [t.strip() for t in p.split(",")]:
            if t and t.lower() not in seen_set:
                seen.append(t)
                seen_set.add(t.lower())
    return ", ".join(seen)


def build_prompt(
    family_label,
    subject_label,
    character_name,
    custom_subject,
    pose_label,
    action_label,
    shot_label,
    angle_label,
    outfit_label,
    location_label,
    time_label,
    lighting_label,
    expression_label,
    style_label,
    quality_tier,
    negative_tier,
    year_tag,
    source_tag,
    rating_tag,
    extra_positive,
    extra_negative,
    char1_traits="",
    char2_traits="",
    char3_traits="",
    char4_traits="",
    use_break=False,
    body_label=NONE,
    intimacy_label="(safe - none)",
    artist_label="(none - use only my visual style preset)",
    artist_strength=1.0,
    custom_artist_tag="",
    force_female_only=True,
    yuri=False,
) -> Tuple[str, str, str]:
    family_label, fam = _resolve_family(family_label)

    subject_t   = _tag(SUBJECTS, subject_label)
    n_chars     = _count_from_subject(subject_label)
    # Yuri tag goes into the subject head when applicable (multi-female).
    if yuri and _is_multi_female(subject_label) and subject_t:
        subject_t = subject_t + ", " + YURI_POSITIVE
    names_list  = _parse_char_names(character_name or "")
    custom_subj = (custom_subject or "").strip().strip(",")

    per_char_all = [
        (char1_traits or "").strip().strip(","),
        (char2_traits or "").strip().strip(","),
        (char3_traits or "").strip().strip(","),
        (char4_traits or "").strip().strip(","),
    ]
    relevant_traits = [t for t in per_char_all[:max(n_chars, 1)] if t]

    pose_t   = _tag(POSES, pose_label)
    action_t = _tag(ACTIONS, action_label)
    shot_t   = _tag(SHOT_TYPES, shot_label)
    angle_t  = _tag(CAMERA_ANGLES, angle_label)
    outfit_t = _tag(OUTFIT_PRESETS, outfit_label)
    loc_t    = _tag(LOCATIONS, location_label)
    time_t   = _tag(TIME_OF_DAY, time_label)
    light_t  = _tag(LIGHTING, lighting_label)
    expr_t     = _tag(EXPRESSIONS, expression_label)
    body_t     = _tag(BODY_TYPES, body_label)
    # Custom pasted tag (from the Explorer) takes priority over the dropdown.
    _custom_t = (custom_artist_tag or "").strip().strip(",")
    if _custom_t:
        _s = float(artist_strength or 1.0)
        if abs(_s - 1.0) < 0.01:
            artist_t = _custom_t
        else:
            artist_t = "({}:{:.2f})".format(_custom_t, _s)
    else:
        artist_t = _artist_emit(artist_label, float(artist_strength or 1.0))
    style_t    = _tag(fam["art_styles"], style_label)
    intimacy_t = INTIMACY_LEVELS.get(intimacy_label, "") if intimacy_label else ""

    meta_parts = []
    if year_tag and year_tag != "(no year tag)": meta_parts.append(year_tag)
    if source_tag and source_tag != "(no source tag)": meta_parts.append(source_tag)
    if rating_tag and rating_tag != "(no rating tag)": meta_parts.append(rating_tag)
    meta_t = ", ".join(meta_parts)

    quality_t   = fam["positive_tiers"].get(quality_tier, "")
    extra_pos_t = (extra_positive or "").strip().strip(",")

    # The "rest" of the prompt (everything after the subject/character clusters)
    # is always deduped via _join.
    rest = _join(
        outfit_t,
        pose_t, action_t,
        shot_t, angle_t,
        loc_t, time_t, light_t,
        expr_t,
        body_t,
        artist_t,
        style_t,
        intimacy_t,
        meta_t, quality_t,
        extra_pos_t,
    )

    # Build per-character clusters: each cluster = name_i + traits_i (either may be
    # empty). Cluster i exists if EITHER the name OR the traits slot has content.
    char_clusters = []
    for i in range(n_chars):
        nm = names_list[i] if i < len(names_list) else ""
        tr = per_char_all[i] if i < len(per_char_all) else ""
        parts = []
        if nm: parts.append(nm)
        if tr: parts.append(tr)
        if parts: char_clusters.append(", ".join(parts))

    if n_chars >= 2 and char_clusters:
        # Multi-character path: each character gets their own cluster so the
        # model can associate that character's name + traits to one person.
        head_parts = [subject_t]
        if custom_subj: head_parts.append(custom_subj)
        head = ", ".join(p for p in head_parts if p)

        if use_break:
            # ' BREAK ' marker chunks the prompt for Forge's attention engine,
            # giving Illustrious / NoobAI cleaner trait separation per character.
            positive = " BREAK ".join([head] + char_clusters)
            if rest:
                positive = positive + ", " + rest
        else:
            # No-BREAK: just concat clusters with commas, no cross-cluster dedup
            all_parts = [head] + char_clusters + ([rest] if rest else [])
            positive  = ", ".join(p for p in all_parts if p)
    else:
        # Solo / no per-char content: collapse all names into the head, dedup
        # everything together.
        solo_name = ", ".join(names_list) if names_list else ""
        positive = _join(subject_t, solo_name, custom_subj, rest)

    neg_t       = fam["negative_tiers"].get(negative_tier, "")
    extra_neg_t = (extra_negative or "").strip().strip(",")
    # Female-only enforcement: if subject is all-female and the checkbox is on,
    # add male-exclusion tags to the negative so Illustrious can't slip a boy in.
    female_only_t = (FEMALE_ONLY_NEGATIVE
                     if (force_female_only and _is_all_female(subject_label))
                     else "")
    negative    = _join(neg_t, female_only_t, extra_neg_t)

    if not positive:
        return ("", "", "Fill in a Subject, Outfit, or Scenario above, then click Build.")
    msg = "Built for {} ({} character{}). Quality: {}. Negative: {}.".format(
        family_label, n_chars, "s" if n_chars != 1 else "",
        quality_tier, negative_tier)
    return positive, negative, msg


# ===========================================================================
# Gradio UI
# ===========================================================================

# ===========================================================================
# ControlNet recipe builder
# ===========================================================================
# Forge stores ControlNet models in models/controlnet/ relative to the Forge
# install root. We scan that folder once at import time, classify each model
# file by name, and use that to populate the recipe UI.

CN_TYPE_KEYWORDS = {
    "canny":     ["canny"],
    "depth":     ["depth"],
    "openpose":  ["openpose", "pose"],
    "lineart":   ["lineart", "line_art", "line-art"],
    "scribble":  ["scribble"],
    "tile":      ["tile"],
    "union":     ["union", "promax"],   # xinsir union/promax all-in-one
    "ipadapter": ["ip-adapter", "ip_adapter", "ipadapter"],
}

CN_INSTALLED = {k: [] for k in CN_TYPE_KEYWORDS}


def _scan_controlnet_models():
    """Find SDXL CN models on disk and group them by type. Idempotent."""
    import os
    global CN_INSTALLED

    # Forge installs at <forge_root>/models/controlnet/ — derive from this
    # script path: scripts/ -> extension/ -> extensions/ -> <forge_root>.
    here = os.path.dirname(os.path.abspath(__file__))
    forge_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    search_paths = [
        os.path.join(forge_root, "models", "controlnet"),
        os.path.join(forge_root, "models", "ControlNet"),
    ]

    # F:/Data/Models is the common StabilityMatrix shared model dir — the user
    # has IP-Adapter SDXL files there too. We scan it as a secondary source.
    extra = os.environ.get("PE_EXTRA_CN_PATH", "")
    if extra:
        search_paths.append(extra)
    for guess in [r"F:\Data\Models\IpAdapter", r"F:\Data\Models\IpAdaptersXl",
                  r"F:\Data\Models\ControlNet"]:
        if os.path.isdir(guess):
            search_paths.append(guess)

    seen_files = set()
    for cat in CN_INSTALLED:
        CN_INSTALLED[cat] = []

    for p in search_paths:
        if not os.path.isdir(p):
            continue
        # Walk into subdirs because StabilityMatrix sets up symlinked subfolders
        # like models/controlnet/IpAdapter -> F:/Data/Models/IpAdapter/. Limit
        # depth so we don't drift into a clip-vision/preprocessor jungle.
        try:
            for dirpath, dirnames, filenames in os.walk(p, followlinks=True):
                # Skip preprocessor caches that pile up large unrelated weights
                low_dir = os.path.basename(dirpath).lower()
                if low_dir in {"preprocessor", "annotator", "clip_vision",
                               "controlnetpreprocessor", "annotators"}:
                    dirnames[:] = []
                    continue
                rel = os.path.relpath(dirpath, p)
                if rel != "." and rel.count(os.sep) > 2:
                    dirnames[:] = []
                    continue
                for fn in filenames:
                    low = fn.lower()
                    if not (low.endswith(".safetensors") or low.endswith(".bin")
                            or low.endswith(".pth") or low.endswith(".ckpt")):
                        continue
                    if fn in seen_files:
                        continue
                    seen_files.add(fn)
                    if low.startswith("put_") or low.startswith("readme"):
                        continue
                    for cat, kws in CN_TYPE_KEYWORDS.items():
                        if any(kw in low for kw in kws):
                            CN_INSTALLED[cat].append(fn)
                            break
                    else:
                        if "diffusion_pytorch_model" in low or "controlnet" in low:
                            CN_INSTALLED["union"].append(fn)
        except Exception as e:
            logger.warning("{} CN scan failed for {}: {}".format(TAG, p, e))

    # Stable sort: preference-friendly names first
    def _rank(fn):
        low = fn.lower()
        score = 0
        for pref in ["diffusers_xl_", "xinsir_", "kohya_controllllite_",
                     "ip-adapter-plus_sdxl_vit-h", "ip-adapter-plus",
                     "thibaud_", "controlnet-"]:
            if low.startswith(pref): score -= 10
            if pref in low:          score -= 5
        # Promax/Union all-in-one is the most useful
        if "promax" in low: score -= 20
        return score
    for cat in CN_INSTALLED:
        CN_INSTALLED[cat] = sorted(CN_INSTALLED[cat], key=_rank)


_scan_controlnet_models()
logger.info("{} controlnet scan: {}".format(
    TAG, {k: len(v) for k, v in CN_INSTALLED.items() if v}))


def _cn_pick(cat, fallback=None):
    """Best installed model name for a given category, or fallback string."""
    lst = CN_INSTALLED.get(cat, [])
    return lst[0] if lst else (fallback or "")


def _import_forge_cn_external_code():
    """Find Forge's ControlNet external_code module. Returns the module or None.
    Forge has moved this around between versions, so we try every known path."""
    candidates = [
        "lib_controlnet.external_code",
        "sd_forge_controlnet.lib_controlnet.external_code",
        "extensions-builtin.sd_forge_controlnet.lib_controlnet.external_code",
        "modules_forge.controlnet",
    ]
    for path in candidates:
        try:
            import importlib
            mod = importlib.import_module(path)
            if hasattr(mod, "ControlNetUnit") or hasattr(mod, "update_cn_script_in_processing"):
                return mod
        except Exception:
            continue
    # Last resort: walk sys.modules for an already-loaded one
    import sys as _sys
    for name, mod in list(_sys.modules.items()):
        if mod is None: continue
        if "external_code" in name and "controlnet" in name.lower():
            return mod
    return None


def _build_cn_units(source_pil, enabled_units_spec):
    """Build a list of ControlNet Unit objects from the user's per-unit spec.
    enabled_units_spec is a list of dicts: {module, model, weight, guidance_end, etc.}
    Each unit gets the same source_pil attached as its image.

    Returns (units_list, error_str). units_list is the format Forge's
    external_code expects; None on error.
    """
    if source_pil is None:
        return None, "No source image uploaded."
    if not enabled_units_spec:
        return None, "No ControlNet units enabled."

    mod = _import_forge_cn_external_code()
    if mod is None:
        return None, ("Forge's ControlNet external_code module not found. "
                      "Make sure the built-in sd_forge_controlnet extension "
                      "is enabled.")

    if not hasattr(mod, "ControlNetUnit"):
        return None, ("ControlNetUnit class not present in {} - your Forge "
                      "build may be too old.".format(mod.__name__))

    import numpy as np
    # Many Forge versions accept PIL directly; some want numpy. We try PIL.
    # If that errors at runtime we'll fall back to numpy below.
    units = []
    for spec in enabled_units_spec:
        try:
            unit = mod.ControlNetUnit(
                enabled=True,
                image=source_pil,
                module=spec.get("module", "none"),
                model=spec.get("model", ""),
                weight=float(spec.get("weight", 1.0)),
                resize_mode=spec.get("resize_mode", "Crop and Resize"),
                low_vram=False,
                processor_res=int(spec.get("processor_res", 1024)),
                guidance_start=float(spec.get("guidance_start", 0.0)),
                guidance_end=float(spec.get("guidance_end", 1.0)),
                pixel_perfect=True,
                control_mode=spec.get("control_mode", "Balanced"),
            )
        except TypeError:
            # Newer Forge ControlNetUnit may use different keyword names.
            # Try a stripped-down call.
            try:
                unit = mod.ControlNetUnit(
                    enabled=True,
                    image=source_pil,
                    module=spec.get("module", "none"),
                    model=spec.get("model", ""),
                    weight=float(spec.get("weight", 1.0)),
                )
            except Exception as e:
                return None, "ControlNetUnit construction failed: {}".format(e)
        units.append(unit)
    return units, None


def _attach_cn_units(p, units, mod):
    """Wire the units into a processing object. Tries multiple Forge APIs."""
    if mod is not None and hasattr(mod, "update_cn_script_in_processing"):
        try:
            mod.update_cn_script_in_processing(p, units)
            return True, None
        except Exception as e:
            return False, "update_cn_script_in_processing failed: {}".format(e)
    # Fallback: set script_args directly.
    try:
        existing = list(getattr(p, "script_args", []) or [])
        # ControlNet expects its args as a flat list in script_args. Convention:
        # script_args[0] is often a stateful marker; we just append the units.
        p.script_args = (*existing, *units)
        return True, None
    except Exception as e:
        return False, "script_args fallback failed: {}".format(e)


# SDXL training resolution buckets. Each entry is (width, height). Both base
# and most fine-tunes (Illustrious, NoobAI, Pony, etc.) were trained on this
# bucket set, so generating at one of these gives the cleanest output.
SDXL_BUCKETS = [
    (1024, 1024),  # 1:1 square
    (1152, 896),   # 1.29:1 landscape
    (896, 1152),   # 0.78:1 portrait
    (1216, 832),   # 1.46:1 landscape
    (832, 1216),   # 0.68:1 portrait  (NAI default)
    (1344, 768),   # 1.75:1 landscape (16:9-ish)
    (768, 1344),   # 0.57:1 portrait
    (1536, 640),   # 2.4:1 wide
    (640, 1536),   # 0.42:1 tall
]


def _snap_to_8(n):
    """SDXL latents need dimensions divisible by 8."""
    return int(round(n / 8) * 8)


def _snap_to_sdxl_bucket(w, h):
    """Pick the SDXL bucket closest to the source aspect ratio.
    Returns (bucket_w, bucket_h). Ties broken by total pixel count (smaller wins)."""
    if w <= 0 or h <= 0:
        return 1024, 1024
    src_aspect = w / float(h)
    def _score(bucket):
        bw, bh = bucket
        ba = bw / float(bh)
        # Compare aspect ratio in log space so 16:9 vs 9:16 are symmetric
        import math
        return abs(math.log(ba) - math.log(src_aspect))
    return min(SDXL_BUCKETS, key=_score)


def _compute_output_size(mode_label, source_pil, slider_w, slider_h,
                          max_dim=1536):
    """Return (width, height) based on the chosen resolution mode."""
    if source_pil is None:
        return int(slider_w), int(slider_h)

    if mode_label and "Snap" in mode_label:
        sw, sh = source_pil.size
        bw, bh = _snap_to_sdxl_bucket(sw, sh)
        return bw, bh

    if mode_label and ("Match" in mode_label or "Source" in mode_label):
        sw, sh = source_pil.size
        # Cap absurdly large source images to max_dim while keeping aspect ratio
        if max(sw, sh) > max_dim:
            scale = max_dim / float(max(sw, sh))
            sw = int(sw * scale); sh = int(sh * scale)
        return _snap_to_8(sw), _snap_to_8(sh)

    # Custom mode: use the sliders verbatim
    return int(slider_w), int(slider_h)


def _run_replication_generate(source_pil, positive, negative, units_spec,
                               steps=28, sampler="Euler a", cfg=6.0,
                               width=1024, height=1024, seed=-1,
                               mode="img2img", denoise=0.55):
    """Top-level replication generator. Returns (PIL.Image|None, info_str).
    mode = 'txt2img' (pure noise start) or 'img2img' (source as init image)."""
    if source_pil is None:
        return None, "No source image."

    try:
        from modules import processing, shared
    except Exception as e:
        return None, "Forge processing module not importable: {}".format(e)

    if not getattr(shared, "sd_model", None):
        return None, ("No checkpoint loaded. Open the txt2img tab once "
                      "to load a model, then come back.")

    units, err = _build_cn_units(source_pil, units_spec)
    if err:
        return None, err

    mod = _import_forge_cn_external_code()

    try:
        if mode == "img2img":
            # img2img path: source is the init image, lower denoise = closer to source
            p = processing.StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                outpath_samples=shared.opts.outdir_img2img_samples,
                outpath_grids=shared.opts.outdir_img2img_grids,
                prompt=positive,
                negative_prompt=negative,
                steps=int(steps),
                sampler_name=str(sampler),
                cfg_scale=float(cfg),
                width=int(width),
                height=int(height),
                seed=int(seed),
                n_iter=1,
                batch_size=1,
                init_images=[source_pil],
                denoising_strength=float(denoise),
                do_not_save_samples=False,
                do_not_save_grid=True,
            )
        else:
            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                outpath_samples=shared.opts.outdir_txt2img_samples,
                outpath_grids=shared.opts.outdir_txt2img_grids,
                prompt=positive,
                negative_prompt=negative,
                steps=int(steps),
                sampler_name=str(sampler),
                cfg_scale=float(cfg),
                width=int(width),
                height=int(height),
                seed=int(seed),
                n_iter=1,
                batch_size=1,
                do_not_save_samples=False,
                do_not_save_grid=True,
            )

        ok, attach_err = _attach_cn_units(p, units, mod)
        if not ok:
            return None, attach_err

        # ---- Inject ControlNet metadata into the saved PNG's infotext block ----
        # Without this the saved image's "parameters" chunk has no CN config,
        # which means re-dragging it back into txt2img would NOT rebuild the
        # CN units. Forge expects per-unit keys: "ControlNet 0", "ControlNet 1",
        # each value being a comma-joined `key: value` string.
        try:
            if not hasattr(p, "extra_generation_params") or p.extra_generation_params is None:
                p.extra_generation_params = {}
            for i, u in enumerate(units_spec):
                if not u.get("model"):
                    continue
                proc_res = int(u.get("processor_res", max(int(width), int(height), 1024)))
                parts = [
                    'Module: "{}"'.format(u.get("module", "none")),
                    'Model: "{}"'.format(u["model"]),
                    "Weight: {:.2f}".format(float(u.get("weight", 1.0))),
                    'Resize Mode: "{}"'.format(u.get("resize_mode", "Crop and Resize")),
                    "Low Vram: False",
                    "Processor Res: {}".format(proc_res),
                    "Guidance Start: {:.2f}".format(float(u.get("guidance_start", 0.0))),
                    "Guidance End: {:.2f}".format(float(u.get("guidance_end", 1.0))),
                    "Pixel Perfect: True",
                    'Control Mode: "{}"'.format(u.get("control_mode", "Balanced")),
                ]
                p.extra_generation_params["ControlNet {}".format(i)] = ", ".join(parts)
        except Exception:
            # Infotext is a nice-to-have; never fail the generation if it errors.
            pass

        processed = processing.process_images(p)
        if not processed or not processed.images:
            return None, "No image returned by processing."
        return processed.images[0], (processed.info or "")
    except Exception as e:
        import traceback
        return None, "Generation failed: {}\n{}".format(e, traceback.format_exc()[-500:])


def _build_controlnet_infotext(units):
    """Format a list of CN unit dicts as Forge infotext lines.
    Each dict: {module, model, weight, guidance_start, guidance_end,
                resize_mode, control_mode, pixel_perfect}.
    """
    out = []
    for i, u in enumerate(units):
        if not u.get("model"):
            continue
        parts = [
            'Module: "{}"'.format(u.get("module", "none")),
            'Model: "{}"'.format(u["model"]),
            "Weight: {:.2f}".format(float(u.get("weight", 1.0))),
            'Resize Mode: "{}"'.format(u.get("resize_mode", "Crop and Resize")),
            "Low Vram: False",
            'Processor Res: 1024',
            "Guidance Start: {:.2f}".format(float(u.get("guidance_start", 0.0))),
            "Guidance End:   {:.2f}".format(float(u.get("guidance_end", 1.0))),
            'Pixel Perfect: True',
            'Control Mode: "{}"'.format(u.get("control_mode", "Balanced")),
        ]
        out.append("ControlNet {}: \"{}\"".format(i, ", ".join(parts)))
    return ", ".join(out) if out else ""


def _on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Prompt Enhancer — guided builder\n"
            "Pick from the dropdowns below. Each option maps to the booru tags "
            "Illustrious and NoobAI XL were trained on — you don't need to know "
            "what 'cowboy shot' or 'rim lighting' means as a tag. Start with a "
            "**Scenario template** below to fill in many sections at once, then tweak."
        )

        with gr.Row():
            with gr.Column(scale=2):
                detected_label = gr.Markdown(
                    "**Detected**: (pick a scenario or click Detect)")
                family = gr.Dropdown(
                    label="Model family",
                    choices=list(MODELS.keys()),
                    value="(detect from loaded checkpoint)",
                    interactive=True,
                )
            with gr.Column(scale=1):
                detect_btn = gr.Button(
                    "\U0001F50D Detect from current checkpoint")

        with gr.Row():
            scenario = gr.Dropdown(
                label="Scenario template (one-click preset — fills the sections below)",
                choices=list(SCENARIOS.keys()),
                value="(none — design from scratch)",
            )
            apply_scenario_btn = gr.Button("Apply scenario", variant="secondary")

        with gr.Accordion("🖼️  Image to Prompt — analyze a reference image",
                          open=False):
            gr.Markdown(
                "Upload an image (your own art, a reference, a screenshot — "
                "anything). The **WD14 v3 tagger** (the Danbooru tagger family "
                "that trained Illustrious / NoobAI) extracts booru-format "
                "tags directly from the pixels.\n\n"
                "**For real style replication, use 'Best (EVA02 large)'** "
                "(default). The base ViT model is faster but its artist "
                "detection is shaky. Detected artist + style tags are "
                "emitted with weight emphasis like `(wlop:1.20)` so the "
                "latent is actually pulled toward that style.\n\n"
                "**First use of each model downloads it (~360 MB - ~1.3 GB, "
                "one-time).** Subsequent analyses are instant."
            )

            with gr.Row():
                analyze_image_input = gr.Image(
                    label="Reference image",
                    type="pil",
                    height=380,
                )
                with gr.Column():
                    wd14_general_thresh = gr.Slider(
                        label="General-tag threshold (lower = more tags)",
                        minimum=0.15, maximum=0.60, step=0.01, value=0.35,
                    )
                    wd14_char_thresh = gr.Slider(
                        label="Character-tag threshold — lower = more candidates surface (recommended 0.55)",
                        minimum=0.30, maximum=0.95, step=0.01, value=0.55,
                    )
                    wd14_artist_thresh = gr.Slider(
                        label="Artist-tag threshold (lower = catches more artist guesses)",
                        minimum=0.15, maximum=0.70, step=0.01, value=0.30,
                    )
                    wd14_model_dd = gr.Dropdown(
                        label="WD14 model — bigger = better artist + style detection",
                        choices=list(WD14_MODELS.keys()),
                        value=WD14_DEFAULT_MODEL_KEY,
                    )
                    artist_weight_slider = gr.Slider(
                        label="🎨 Artist tag weight (>1.0 forces style harder)",
                        minimum=0.5, maximum=1.5, step=0.05, value=1.20,
                    )
                    style_weight_slider = gr.Slider(
                        label="🎨 Style descriptor weight (>1.0 forces medium harder)",
                        minimum=0.5, maximum=1.5, step=0.05, value=1.15,
                    )
                    add_quality_cb = gr.Checkbox(
                        label="Wrap tags with Illustrious quality + standard negative",
                        value=True,
                    )
                    analyze_btn = gr.Button(
                        "🔍  Analyze image", variant="primary", size="lg",
                    )

            analyze_status = gr.Markdown(
                value="*Upload an image, set the thresholds, then click Analyze.*"
            )

            extracted_chars_box = gr.Textbox(
                label="🧑 Detected character(s) — primary + others + metadata cross-check",
                lines=4,
                show_copy_button=True,
            )
            extracted_artist_box = gr.Textbox(
                label="Detected artist style(s) + medium / rendering style",
                lines=2,
                show_copy_button=True,
            )
            extracted_tags_box = gr.Textbox(
                label="Extracted tags (1:1 replication prompt)",
                lines=5,
                show_copy_button=True,
            )
            extracted_scene_box = gr.Textbox(
                label="🌍 Scene context — Time / Weather / Lighting / Location / Season / Sky",
                lines=4,
                show_copy_button=True,
            )
            extracted_rating_box = gr.Textbox(
                label="Content rating (Danbooru)",
                lines=1,
            )

            gr.Markdown("---")
            gr.Markdown("### 📜 Stage 1: Metadata embedded in the image")
            extracted_meta_source = gr.Textbox(
                label="Metadata source (A1111 / Forge / ComfyUI / EXIF / none)",
                lines=1,
            )
            extracted_meta_positive = gr.Textbox(
                label="Original positive prompt (if image is AI-generated)",
                lines=4,
                show_copy_button=True,
            )
            extracted_meta_negative = gr.Textbox(
                label="Original negative prompt",
                lines=2,
                show_copy_button=True,
            )
            extracted_meta_settings = gr.Textbox(
                label="Original generation settings (steps, sampler, CFG, seed, size, model)",
                lines=3,
                show_copy_button=True,
            )

            gr.Markdown("---")
            gr.Markdown(
                "### 🪄 Stage 3: Ultimate Prompt — metadata first, then WD14 "
                "artist (weighted) + characters + scene + style + general tags "
                "+ quality boilerplate. Deduped case-insensitively."
            )
            ultimate_prompt_box = gr.Textbox(
                label="✨ Ultimate positive prompt (use this for replication)",
                lines=6,
                show_copy_button=True,
            )
            ultimate_negative_box = gr.Textbox(
                label="✨ Ultimate negative prompt",
                lines=3,
                show_copy_button=True,
            )
            ultimate_info_box = gr.Markdown(
                value="*Click **Analyze image** to populate the ultimate "
                      "prompt. The 'Generate replication NOW' button below "
                      "uses this.*"
            )

            with gr.Row():
                insert_into_subject_btn = gr.Button(
                    "↩️  Insert WD14 tags into 'Extra subject details' above",
                    scale=1,
                )
                replicate_t2i_btn = gr.Button(
                    "🎯  Replicate (send to txt2img)",
                    variant="primary",
                    scale=1,
                )

            # Hidden state holds the formatted-for-paste prompt string.
            replicate_params_state = gr.Textbox(visible=False)

            gr.Markdown("---")
            gr.Markdown(
                "### 🎮 ControlNet recipe (for closer 1:1 replication)\n"
                "Tags alone get you ~70% of the way. **ControlNet locks in "
                "composition, pose, and outlines from the reference image — "
                "the remaining 30%.** Tick the units you want, click the "
                "**Send to txt2img with ControlNet preset** button below, "
                "then in txt2img drop the **same source image** into each "
                "enabled unit's image input (Forge's paste API can't transfer "
                "images across tabs — only the model + weights). Hit Generate."
            )

            # Render a short installed-models summary so the user sees what's available
            _installed_md_lines = ["**Detected on your install:**"]
            for cat_label, key in [
                ("Canny", "canny"), ("Depth", "depth"),
                ("OpenPose", "openpose"), ("Lineart", "lineart"),
                ("Tile", "tile"), ("Union/Promax (all-in-one)", "union"),
                ("IP-Adapter (style transfer)", "ipadapter"),
            ]:
                lst = CN_INSTALLED.get(key, [])
                if lst:
                    _installed_md_lines.append(
                        "- **{}**: `{}`{}".format(
                            cat_label, lst[0],
                            "" if len(lst) == 1 else " (+{} more)".format(len(lst) - 1)
                        ))
                else:
                    _installed_md_lines.append("- {} — *(none installed)*".format(cat_label))
            gr.Markdown("\n".join(_installed_md_lines))

            replicate_mode_radio = gr.Radio(
                label="Replication mode",
                choices=["txt2img (creative — generate from prompt + CN)",
                         "img2img (1:1 — starts from your source image)"],
                value="img2img (1:1 — starts from your source image)",
            )
            replicate_denoise = gr.Slider(
                label="Denoise (img2img only) — lower = closer to source",
                minimum=0.20, maximum=0.95, step=0.02, value=0.55,
            )
            replicate_intensity = gr.Radio(
                label="🎯 Replication intensity preset (tunes all the weights below)",
                choices=[
                    "Loose (creative variation)",
                    "Balanced (closer match)",
                    "Strict (near-1:1) - recommended",
                    "Maximum (almost identical)",
                ],
                value="Strict (near-1:1) - recommended",
            )

            with gr.Row():
                cn_canny_cb = gr.Checkbox(
                    label="Canny edges (structure lock)", value=True,
                    interactive=bool(CN_INSTALLED["canny"] or CN_INSTALLED["union"]),
                )
                cn_canny_weight = gr.Slider(label="weight", minimum=0.0, maximum=1.5, step=0.05, value=0.85)
                cn_canny_end = gr.Slider(label="end step", minimum=0.0, maximum=1.0, step=0.05, value=0.70)

            with gr.Row():
                cn_depth_cb = gr.Checkbox(
                    label="Depth (3D layout)", value=True,
                    interactive=bool(CN_INSTALLED["depth"] or CN_INSTALLED["union"]),
                )
                cn_depth_weight = gr.Slider(label="weight", minimum=0.0, maximum=1.5, step=0.05, value=0.65)
                cn_depth_end = gr.Slider(label="end step", minimum=0.0, maximum=1.0, step=0.05, value=0.60)

            with gr.Row():
                cn_pose_cb = gr.Checkbox(
                    label="OpenPose (character pose)", value=False,
                    interactive=bool(CN_INSTALLED["openpose"] or CN_INSTALLED["union"]),
                )
                cn_pose_weight = gr.Slider(label="weight", minimum=0.0, maximum=1.5, step=0.05, value=0.80)
                cn_pose_end = gr.Slider(label="end step", minimum=0.0, maximum=1.0, step=0.05, value=0.80)

            with gr.Row():
                cn_tile_cb = gr.Checkbox(
                    label="Tile (preserves colors + coarse layout — great for upscaling/refine)",
                    value=False,
                    interactive=bool(CN_INSTALLED["tile"]),
                )
                cn_tile_weight = gr.Slider(label="weight", minimum=0.0, maximum=1.5, step=0.05, value=0.50)
                cn_tile_end = gr.Slider(label="end step", minimum=0.0, maximum=1.0, step=0.05, value=0.80)

            with gr.Row():
                cn_ipadapter_cb = gr.Checkbox(
                    label="🎨 IP-Adapter Plus (style / image transfer — KEY for art style replication)",
                    value=True,
                    interactive=bool(CN_INSTALLED["ipadapter"]),
                )
                cn_ipadapter_weight = gr.Slider(label="weight", minimum=0.0, maximum=1.5, step=0.05, value=0.90)
                cn_ipadapter_end = gr.Slider(label="end step", minimum=0.0, maximum=1.0, step=0.05, value=1.00)

            replicate_resolution_mode = gr.Radio(
                label="Output size — match source dimensions or snap to SDXL bucket",
                choices=[
                    "Match source image dimensions",
                    "Snap to nearest SDXL training bucket (recommended)",
                    "Custom (use Width/Height sliders below)",
                ],
                value="Snap to nearest SDXL training bucket (recommended)",
            )

            with gr.Accordion("Generation settings (only for the direct-generate button)",
                              open=False):
                with gr.Row():
                    cn_steps = gr.Slider(label="Steps", minimum=10, maximum=60, step=1, value=28)
                    cn_cfg   = gr.Slider(label="CFG", minimum=2.0, maximum=12.0, step=0.5, value=6.0)
                with gr.Row():
                    cn_sampler = gr.Dropdown(
                        label="Sampler",
                        choices=["Euler a", "Euler", "DPM++ 2M",
                                 "DPM++ 2M Karras", "DPM++ SDE",
                                 "DPM++ 3M SDE", "UniPC", "LCM"],
                        value="Euler a",
                    )
                    cn_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                with gr.Row():
                    cn_width  = gr.Slider(label="Width",  minimum=512, maximum=1536, step=64, value=1024)
                    cn_height = gr.Slider(label="Height", minimum=512, maximum=1536, step=64, value=1024)

            with gr.Row():
                generate_cn_btn = gr.Button(
                    "🎯 Generate replication NOW (auto-attaches image to CN units)",
                    variant="primary", size="lg",
                )

            cn_status = gr.Markdown(
                value="*This button bypasses the txt2img tab and runs the generation "
                      "directly with ControlNet pre-configured + your reference image "
                      "already attached to every enabled unit. Result appears below.*"
            )

            cn_result_image = gr.Image(
                label="Replication result",
                interactive=False,
                show_download_button=True,
                height=520,
            )

        with gr.Accordion("1. Subject / Character", open=True):
            subject_dd = gr.Dropdown(
                label="Subject (count + gender combined — no conflicts possible)",
                choices=list(SUBJECTS.keys()),
                value="Solo - 1girl (young adult female)",
            )
            character_name = gr.Textbox(
                label="Character name(s) — for multi-character, separate with commas; "
                      "one name per character (paren-aware so series tags survive)",
                placeholder="Solo: seele_vollerei  |  Trio: "
                            "kafka_(honkai:_star_rail), seele_vollerei, bronya_rand",
            )
            custom_subject = gr.Textbox(
                label="Extra subject details — SHARED across all characters "
                      "(hair, eyes, accessories that apply to everyone)",
                placeholder="e.g. blue hair, blue eyes, long hair, ahoge, glasses",
                lines=2,
            )

            with gr.Row():
                force_female_only_cb = gr.Checkbox(
                    label="🚫 Force female-only (adds male-exclusion to negative "
                          "— recommended for any all-female subject so the model "
                          "can't slip a boy in)",
                    value=True,
                )
                yuri_cb = gr.Checkbox(
                    label="💕 Add 'yuri' tag (only meaningful with 2+ females — "
                          "use for romantic / intimate same-sex context)",
                    value=False,
                )

        with gr.Accordion("1b. Per-character traits (only used when 2+ characters)",
                          open=False):
            gr.Markdown(
                "Each box describes ONE character's unique traits. Only the first "
                "N boxes are used based on your Subject choice (Pair=2, Trio=3, "
                "Group=up to 4). Leave blank for any character that should share "
                "the defaults. Solo subjects ignore this section."
            )
            with gr.Row():
                char1_traits = gr.Textbox(
                    label="Character 1 traits",
                    placeholder="e.g. blue hair, blue eyes, school uniform, twintails",
                    lines=2,
                )
                char2_traits = gr.Textbox(
                    label="Character 2 traits",
                    placeholder="e.g. red hair, red eyes, casual clothes, short hair",
                    lines=2,
                )
            with gr.Row():
                char3_traits = gr.Textbox(
                    label="Character 3 traits",
                    placeholder="e.g. black hair, brown eyes, business suit",
                    lines=2,
                )
                char4_traits = gr.Textbox(
                    label="Character 4 traits (Group only)",
                    placeholder="e.g. blonde hair, green eyes, sundress",
                    lines=2,
                )
            use_break = gr.Checkbox(
                label="Use BREAK separator between characters "
                      "(off by default — plain commas tend to work better)",
                value=False,
            )

        with gr.Accordion("2. Pose & Action", open=True):
            with gr.Row():
                pose_dd = gr.Dropdown(
                    label="Pose", choices=list(POSES.keys()), value="Standing")
                action_dd = gr.Dropdown(
                    label="Action / gaze", choices=list(ACTIONS.keys()),
                    value="Looking at viewer")

        with gr.Accordion("3. Framing", open=True):
            with gr.Row():
                shot_dd = gr.Dropdown(
                    label="Shot type", choices=list(SHOT_TYPES.keys()),
                    value="Upper body")
                angle_dd = gr.Dropdown(
                    label="Camera angle", choices=list(CAMERA_ANGLES.keys()),
                    value="Eye level (default)")

        with gr.Accordion("4. Outfit", open=True):
            outfit_dd = gr.Dropdown(
                label="Outfit style", choices=list(OUTFIT_PRESETS.keys()),
                value="Casual everyday")

        with gr.Accordion("5. Scene, Time & Lighting", open=True):
            location_dd = gr.Dropdown(
                label="Location", choices=list(LOCATIONS.keys()),
                value="Simple white background")
            with gr.Row():
                time_dd = gr.Dropdown(
                    label="Time of day", choices=list(TIME_OF_DAY.keys()),
                    value="Afternoon")
                lighting_dd = gr.Dropdown(
                    label="Lighting", choices=list(LIGHTING.keys()),
                    value="Soft natural")

        with gr.Accordion("6. Expression & Mood", open=False):
            expression_dd = gr.Dropdown(
                label="Expression", choices=list(EXPRESSIONS.keys()),
                value="Soft smile")

        with gr.Accordion("7. Art Style (model-aware)", open=True):
            style_dd = gr.Dropdown(
                label="Visual style",
                choices=list(ART_STYLES_ILLUSTRIOUS.keys()),
                value="Detailed anime (default)",
            )

        with gr.Accordion("8. Quality, Negative & Booru meta", open=False):
            quality = gr.Radio(
                label="Quality tier (positive baseline)",
                choices=list(ILLUSTRIOUS_QUALITY_TIERS.keys()),
                value="Standard (safe default)",
            )
            negative = gr.Radio(
                label="Negative tier",
                choices=list(ILLUSTRIOUS_NEGATIVE_TIERS.keys()),
                value="Standard (safe default)",
            )
            with gr.Row():
                year   = gr.Dropdown(label="Year tag", choices=YEAR_TAGS, value="newest")
                source = gr.Dropdown(label="Source tag", choices=SOURCE_TAGS, value="(no source tag)")
                rating = gr.Dropdown(label="Rating tag", choices=RATING_TAGS, value="(no rating tag)")

        with gr.Accordion("9. Body / Physique (optional)", open=False):
            body_dd = gr.Dropdown(
                label="Body type / physique - mix safe and [NSFW] freely",
                choices=list(BODY_TYPES.keys()),
                value=NONE,
            )

        with gr.Accordion("9b. Artist style — searchable 33k artist gallery",
                          open=False):
            gr.Markdown(
                "### 🎨 Pick an artist from the full Illustrious / NoobAI catalogue\n"
                "Type any part of an artist's name in the **Search** dropdown — "
                "filtering happens as you type. Pick one and a real Danbooru "
                "sample appears below so you can see exactly what their style "
                "looks like before you commit. Use the strength slider to dial "
                "the influence up or down."
            )

            # Searchable dropdown over all 33,719 artists. Gradio's
            # filterable=True gives instant in-place filtering as the user types.
            full_artist_dd = gr.Dropdown(
                label="🔎 Search artists ({:,} total) — type any part of a name".format(
                    len(FULL_ARTIST_ITEMS)),
                choices=FULL_ARTIST_LABELS,
                value="(none)",
                filterable=True,
                interactive=True,
            )

            with gr.Row():
                preview_btn = gr.Button(
                    "🖼️  Fetch sample image from Danbooru",
                    variant="secondary",
                    scale=1,
                )
                use_artist_btn = gr.Button(
                    "✅  Use this artist in my prompt",
                    variant="primary",
                    scale=1,
                )

            preview_image = gr.Image(
                label="Sample (random pick from Danbooru — refresh button = new pick)",
                interactive=False,
                show_download_button=False,
                height=380,
            )
            preview_info = gr.Markdown(
                value="*Pick an artist above, then click "
                      "**Fetch sample image from Danbooru** to see what their "
                      "style looks like.*",
            )

            # The "currently selected" textbox is what build_prompt actually
            # reads. The dropdown writes here when the user clicks
            # "Use this artist", and the user can also paste/edit freely.
            custom_artist_tag = gr.Textbox(
                label="Selected artist tag (used in the prompt — edit freely)",
                placeholder="e.g. wlop, ciloranko, kantoku — or type any "
                            "Danbooru artist tag",
                lines=1,
            )



            gr.Markdown(
                "---\n**Backup quick-picks** (the small curated list, kept "
                "as a fallback). Overridden by the search/textbox above:"
            )
            artist_dd = gr.Dropdown(
                label="Quick-pick artist (curated list)",
                choices=list(ARTIST_STYLES.keys()),
                value="(none - use only my visual style preset)",
            )
            artist_strength = gr.Slider(
                label="Artist strength (1.0 = full, lower = subtler, "
                      "higher = more dominant). Wraps as (artist:strength) "
                      "when not 1.0.",
                minimum=0.3, maximum=1.5, step=0.05, value=1.0,
            )
            artist_preview = gr.Markdown(
                value=_artist_preview_md("(none - use only my visual style preset)"),
            )

            def _refresh_artist_preview(lbl):
                return _artist_preview_md(lbl)

            artist_dd.change(fn=_refresh_artist_preview,
                             inputs=[artist_dd],
                             outputs=[artist_preview])

            # ---- searchable dropdown wiring ----

            def _on_pick_full_artist(label):
                if not label or label == "(none)":
                    return ("*Pick an artist above, then click "
                            "**Fetch sample image from Danbooru** to see what "
                            "their style looks like.*")
                tag = FULL_ARTIST_LABEL_TO_TAG.get(label, "")
                if not tag:
                    return "*Unknown artist label.*"
                url = "https://danbooru.donmai.us/posts?tags={}".format(tag)
                return ("**Selected**: `{}`  \n"
                        "[View all samples on Danbooru]({}) — or click "
                        "**Fetch sample image** to load one inline.").format(tag, url)

            full_artist_dd.change(
                fn=_on_pick_full_artist,
                inputs=[full_artist_dd],
                outputs=[preview_info],
            )

            def _do_fetch_preview(label):
                if not label or label == "(none)":
                    return None, ("*Pick an artist first.*")
                tag = FULL_ARTIST_LABEL_TO_TAG.get(label, "")
                if not tag:
                    return None, "*Unknown artist.*"
                url = _danbooru_preview_url(tag)
                if not url:
                    return None, ("**Selected**: `{}`  \n"
                                  "(Danbooru returned no samples for this tag — "
                                  "the artist might exist only under a slightly "
                                  "different name. Open the link in your "
                                  "browser to confirm.)\n\n"
                                  "[Search Danbooru for {}]"
                                  "(https://danbooru.donmai.us/posts?tags={})"
                                  ).format(tag, tag, tag)
                view = "https://danbooru.donmai.us/posts?tags={}".format(tag)
                return url, ("**Selected**: `{}`  \n"
                             "Sample loaded above. Click "
                             "**Fetch sample image** again for a different "
                             "random pick.  \n"
                             "[See all samples on Danbooru]({})").format(tag, view)

            preview_btn.click(
                fn=_do_fetch_preview,
                inputs=[full_artist_dd],
                outputs=[preview_image, preview_info],
            )

            def _do_use_artist(label, current):
                if not label or label == "(none)":
                    return current
                return FULL_ARTIST_LABEL_TO_TAG.get(label, current)

            use_artist_btn.click(
                fn=_do_use_artist,
                inputs=[full_artist_dd, custom_artist_tag],
                outputs=[custom_artist_tag],
            )

        with gr.Accordion("10. Intimacy / NSFW level", open=False):
            gr.Markdown(
                "Default is **(safe - none)**. Pick a higher tier only if you "
                "want the model to lean into intimate, suggestive, or explicit "
                "content. [NSFW] options require a permissive checkpoint "
                "(Illustrious / NoobAI are fine)."
            )
            intimacy_dd = gr.Dropdown(
                label="Intimacy / explicitness",
                choices=list(INTIMACY_LEVELS.keys()),
                value="(safe - none)",
            )

        with gr.Accordion("11. Custom extras (optional)", open=False):
            extra_positive = gr.Textbox(
                label="Append to POSITIVE",
                placeholder="e.g. cinematic lighting, <lora:my_style:0.7>",
                lines=2,
            )
            extra_negative = gr.Textbox(
                label="Append to NEGATIVE",
                placeholder="e.g. monochrome, censored, multiple views",
                lines=2,
            )

        build_btn = gr.Button("Build prompt →", variant="primary", size="lg")

        positive_out = gr.Textbox(
            label="Generated POSITIVE prompt", lines=8,
            show_copy_button=True,
            placeholder="Click 'Build prompt' to assemble.",
        )
        negative_out = gr.Textbox(
            label="Generated NEGATIVE prompt", lines=5,
            show_copy_button=True,
        )
        status = gr.Textbox(label="Status", interactive=False, lines=2,
                            value="Pick a scenario or fill in the sections, "
                                  "then click 'Build prompt'.")
        with gr.Row():
            send_t2i_btn = gr.Button("Send to txt2img", variant="primary")
            send_i2i_btn = gr.Button("Send to img2img")

        params_state = gr.Textbox(visible=False)

        # ---- Wiring ----

        def _do_detect():
            ckpt = _get_loaded_checkpoint()
            fam_label = _detect_model_family(ckpt)
            md = "**Detected**: `{}` → **{}**".format(
                ckpt or "(no checkpoint loaded)", fam_label)
            styles = list(MODELS[fam_label]["art_styles"].keys())
            return md, gr.update(value=fam_label), gr.update(
                choices=styles, value=styles[0] if styles else None)

        detect_btn.click(fn=_do_detect,
                         outputs=[detected_label, family, style_dd])

        def _on_family_change(family_label):
            _, fam = _resolve_family(family_label)
            styles = list(fam["art_styles"].keys())
            return gr.update(choices=styles, value=styles[0] if styles else None)

        family.change(fn=_on_family_change, inputs=[family], outputs=[style_dd])

        def _apply_scenario(scenario_label):
            tmpl = SCENARIOS.get(scenario_label, {})
            if not tmpl:
                # No template selected — don't touch the form, just hint.
                return [gr.update()] * 11 + [
                    "Pick a scenario template above first, then click Apply."
                ]
            return [
                gr.update(value=tmpl.get("subject", "Solo - 1girl (young adult female)")),
                gr.update(value=tmpl.get("pose",     "Standing")),
                gr.update(value=tmpl.get("action",   "Looking at viewer")),
                gr.update(value=tmpl.get("shot",     "Upper body")),
                gr.update(value=tmpl.get("angle",    "Eye level (default)")),
                gr.update(value=tmpl.get("outfit",   "Casual everyday")),
                gr.update(value=tmpl.get("location", "Simple white background")),
                gr.update(value=tmpl.get("time",     "Afternoon")),
                gr.update(value=tmpl.get("lighting", "Soft natural")),
                gr.update(value=tmpl.get("expr",     "Soft smile")),
                gr.update(value=tmpl.get("style",    "Detailed anime (default)")),
                "Scenario applied: {} - you can still tweak any section.".format(scenario_label),
            ]

        apply_scenario_btn.click(
            fn=_apply_scenario, inputs=[scenario],
            outputs=[subject_dd, pose_dd, action_dd, shot_dd, angle_dd,
                     outfit_dd, location_dd, time_dd, lighting_dd,
                     expression_dd, style_dd, status],
        )

        # ---------- Image-to-Prompt wiring ----------

        def _weighted(tag_list, strength):
            """Apply (tag:strength) wrapping when strength != 1.0."""
            if abs(strength - 1.0) < 0.01:
                return ", ".join([t[0] for t in tag_list])
            return ", ".join(["({}:{:.2f})".format(t[0], strength) for t in tag_list])

        ILLUSTRIOUS_REPLICATE_QUALITY = (
            "masterpiece, best quality, very aesthetic, absurdres, newest"
        )
        ILLUSTRIOUS_REPLICATE_NEGATIVE = (
            "worst quality, low quality, normal quality, lowres, "
            "bad anatomy, bad hands, watermark, signature, jpeg artifacts, "
            "blurry, sketch"
        )

        def _do_analyze_image(img, gen_thr, char_thr, art_thr,
                              model_key, artist_w, style_w, add_quality):
            if img is None:
                # 7 original outputs + 6 new = 13
                return ("*No image uploaded.*", "", "", "", "", "", "",
                        "", "", "", "", "", "")
            try:
                # ---- Stage 1: metadata ----
                meta = _extract_image_metadata(img)

                # ---- Stage 2: WD14 ----
                res = analyze_image_wd14(
                    img,
                    general_threshold=float(gen_thr),
                    character_threshold=float(char_thr),
                    artist_threshold=float(art_thr),
                    model_key=model_key)

                # ---- Build structured character display ----
                wd14_chars = res["characters"]
                all_chars = res.get("characters_all", []) or wd14_chars
                meta_promoted = _scan_metadata_for_characters(
                    meta["positive"], all_chars)
                # Merge: metadata-promoted at top, then WD14 above threshold,
                # de-duplicated by tag.
                seen_char_tags = set()
                char_lines = []
                if meta_promoted:
                    parts = []
                    for tag, sc in meta_promoted:
                        if tag in seen_char_tags: continue
                        seen_char_tags.add(tag)
                        parts.append(_format_character_display(tag, sc))
                    if parts:
                        char_lines.append("**Confirmed from metadata prompt**: " + ", ".join(parts))
                if wd14_chars:
                    primary = None
                    others = []
                    for tag, sc in wd14_chars:
                        if tag in seen_char_tags: continue
                        seen_char_tags.add(tag)
                        if primary is None:
                            primary = (tag, sc)
                        else:
                            others.append((tag, sc))
                    if primary:
                        char_lines.append("**Primary**: " + _format_character_display(*primary))
                    if others:
                        char_lines.append("**Also detected**: " + ", ".join(
                            _format_character_display(t, s) for t, s in others[:6]))
                # Also surface near-misses (just below threshold) for awareness
                near_misses = [
                    (t, s) for (t, s) in all_chars
                    if s < float(char_thr) and s >= max(0.30, float(char_thr) - 0.20)
                    and t not in seen_char_tags
                ][:4]
                if near_misses:
                    char_lines.append("**Near-miss candidates** (raise threshold to include): "
                                       + ", ".join(_format_character_display(t, s)
                                                    for t, s in near_misses))
                chars = "\n".join(char_lines) or "(no character detected — try the Best WD14 model and lower the character threshold)"
                rating = ", ".join(
                    ["{} ({:.0%})".format(t[0], t[1]) for t in res["ratings"][:1]]
                ) or "(unknown)"

                # Apply weight emphasis to artists + style descriptors.
                aw = float(artist_w)
                sw = float(style_w)
                weighted_chars   = _weighted(res["characters"], 1.0)
                weighted_artists = _weighted(res["artists"], aw)
                weighted_general = _weighted(res["general"], 1.0)
                weighted_styles  = _weighted(res["styles"], sw)
                tag_parts = [p for p in [weighted_chars, weighted_artists,
                                          weighted_general, weighted_styles] if p]
                tag_str = ", ".join(tag_parts)

                # Build the human-readable Artist + Style display field.
                artist_lines = []
                if res["artists"]:
                    art_str = ", ".join(
                        ["{} ({:.0%}, weight {:.2f})".format(t[0], t[1], aw)
                         for t in res["artists"][:8]]
                    )
                    artist_lines.append("Artists: " + art_str)
                if res["styles"]:
                    style_str = ", ".join(
                        ["{} (weight {:.2f})".format(t[0], sw)
                         for t in res["styles"]]
                    )
                    artist_lines.append("Style: " + style_str)
                artist_field = "\n".join(artist_lines) or "(no artist or style descriptors detected — try the Best WD14 model and lower the artist threshold)"

                # Build the structured Scene Context display.
                scene_lines = []
                for label in ["Time", "Weather", "Lighting", "Location",
                              "Season", "Sky"]:
                    hits = res["scene_context"].get(label, [])
                    if hits:
                        # Show top 5 with confidence in each bucket
                        items = ", ".join(
                            ["{} ({:.0%})".format(t[0], t[1]) for t in hits[:5]]
                        )
                        scene_lines.append("**{}**: {}".format(label, items))
                scene_field = "\n".join(scene_lines) or "(no scene context detected)"

                # NOTE: an earlier code path built a `paste` from `tag_str` here
                # and immediately threw it away after the Ultimate Prompt block.
                # Removed; the surviving paste is built below using ult_pos/ult_neg.

                # ---- Build metadata display strings ----
                meta_source = meta["source"]
                meta_settings_str = ""
                if meta["settings"]:
                    meta_settings_str = ", ".join(
                        ["{}: {}".format(k, v) for k, v in meta["settings"].items()]
                    )
                # Truncate ridiculously long EXIF dumps for the settings field
                if meta["source"] == "exif" and meta_settings_str:
                    # If EXIF only, show first few keys
                    keys = ["Make", "Model", "Software", "DateTime",
                            "DateTimeOriginal", "ISOSpeedRatings", "FNumber",
                            "ExposureTime", "FocalLength"]
                    parts = []
                    for k in keys:
                        if k in meta["exif"]:
                            parts.append("{}: {}".format(k, meta["exif"][k]))
                    if parts:
                        meta_settings_str = " | ".join(parts)

                # ---- Stage 3: Ultimate Prompt ----
                ult_pos, ult_neg, ult_info = build_ultimate_prompt(
                    meta, res,
                    artist_weight=aw, style_weight=sw,
                    add_quality=bool(add_quality))

                if ult_info.get("used_metadata"):
                    ult_md_status = (
                        "✅ Ultimate prompt built from **{}** metadata + WD14 layers: "
                        + " · ".join(ult_info["layers"])
                    ).format(meta_source)
                else:
                    ult_md_status = (
                        "✅ Ultimate prompt built from WD14 only (no embedded "
                        "metadata in this image). Layers: "
                        + " · ".join(ult_info["layers"])
                    )

                # ---- Replicate paste params now uses the ULTIMATE prompt
                # and the SOURCE-IMAGE-DERIVED size (snapped to nearest SDXL bucket).
                try:
                    _sw, _sh = img.size
                    _pw, _ph = _snap_to_sdxl_bucket(_sw, _sh)
                    paste_size_ult = "{}x{}".format(_pw, _ph)
                except Exception:
                    paste_size_ult = "1024x1024"

                # If the source image embedded a Model hint, preserve it so the
                # paste-button can carry it through to txt2img/img2img.
                _src_model = meta["settings"].get("Model", "") if meta.get("settings") else ""
                _model_suffix = ", Model: {}".format(_src_model) if _src_model else ""

                if add_quality:
                    paste = ("{}\nNegative prompt: {}\nSteps: 28, "
                             "Sampler: Euler a, CFG scale: 6, Size: {}{}"
                             ).format(ult_pos, ult_neg, paste_size_ult,
                                      _model_suffix)
                else:
                    paste = ("{}\nSteps: 28, Sampler: Euler a, "
                             "CFG scale: 6, Size: {}{}").format(
                                ult_pos, paste_size_ult, _model_suffix)

                status_md = ("✅ {} model · {} metadata · "
                             "**{}** general · **{}** artist · **{}** style · "
                             "**{}** character(s) tags from WD14. Artist weight "
                             "**{:.2f}**, style weight **{:.2f}**."
                             ).format(model_key, meta_source,
                                      len(res["general"]),
                                      len(res["artists"]), len(res["styles"]),
                                      len(res["characters"]), aw, sw)
                return (status_md, chars, artist_field, scene_field,
                        tag_str, rating, paste,
                        meta_source, meta["positive"], meta["negative"],
                        meta_settings_str, ult_pos, ult_neg)
            except Exception as e:
                import traceback
                err = "❌ Analysis failed: `{}`\n```\n{}\n```".format(
                    e, traceback.format_exc()[-500:])
                return (err, "", "", "", "", "", "",
                        "", "", "", "", "", "")

        analyze_btn.click(
            fn=_do_analyze_image,
            inputs=[analyze_image_input, wd14_general_thresh,
                    wd14_char_thresh, wd14_artist_thresh,
                    wd14_model_dd, artist_weight_slider,
                    style_weight_slider, add_quality_cb],
            outputs=[analyze_status, extracted_chars_box,
                     extracted_artist_box, extracted_scene_box,
                     extracted_tags_box, extracted_rating_box,
                     replicate_params_state,
                     extracted_meta_source, extracted_meta_positive,
                     extracted_meta_negative, extracted_meta_settings,
                     ultimate_prompt_box, ultimate_negative_box],
        )

        # When a new image is uploaded, auto-set Width/Height to the snapped
        # SDXL-bucket dimensions of the source so the resolution is visually
        # obvious AND propagates to both Generate buttons (the slider values are
        # what each handler reads).
        def _on_image_upload(img):
            if img is None:
                return gr.update(), gr.update()
            try:
                w, h = img.size
                bw, bh = _snap_to_sdxl_bucket(w, h)
                return gr.update(value=bw), gr.update(value=bh)
            except Exception:
                return gr.update(), gr.update()

        analyze_image_input.change(
            fn=_on_image_upload,
            inputs=[analyze_image_input],
            outputs=[cn_width, cn_height],
        )

        def _do_insert_into_subject(tags, current):
            if not tags or not tags.strip():
                return current
            parts = []
            if current and current.strip():
                parts.append(current.strip().rstrip(","))
            parts.append(tags.strip().rstrip(","))
            return ", ".join(parts)

        insert_into_subject_btn.click(
            fn=_do_insert_into_subject,
            inputs=[extracted_tags_box, custom_subject],
            outputs=[custom_subject],
        )

        # Intensity preset -> weight/mode dial.
        INTENSITY_PRESETS = {
            "Loose (creative variation)": {
                "mode_label": "txt2img (creative — generate from prompt + CN)",
                "denoise": None, "cfg": 6.0,
                "canny_w": 0.50, "canny_end": 0.40,
                "depth_w": 0.30, "depth_end": 0.30,
                "pose_w":  0.60, "pose_end":  0.60,
                "tile_w":  0.40, "tile_end":  0.50,
                "ipa_w":   0.50, "ipa_end":   0.80,
            },
            "Balanced (closer match)": {
                "mode_label": "txt2img (creative — generate from prompt + CN)",
                "denoise": None, "cfg": 6.0,
                "canny_w": 0.75, "canny_end": 0.60,
                "depth_w": 0.55, "depth_end": 0.50,
                "pose_w":  0.75, "pose_end":  0.75,
                "tile_w":  0.50, "tile_end":  0.70,
                "ipa_w":   0.75, "ipa_end":   1.00,
            },
            "Strict (near-1:1) - recommended": {
                "mode_label": "img2img (1:1 — starts from your source image)",
                "denoise": 0.55, "cfg": 5.5,
                "canny_w": 0.90, "canny_end": 0.75,
                "depth_w": 0.70, "depth_end": 0.65,
                "pose_w":  0.85, "pose_end":  0.85,
                "tile_w":  0.55, "tile_end":  0.80,
                "ipa_w":   0.95, "ipa_end":   1.00,
            },
            "Maximum (almost identical)": {
                "mode_label": "img2img (1:1 — starts from your source image)",
                "denoise": 0.40, "cfg": 5.0,
                "canny_w": 1.10, "canny_end": 0.90,
                "depth_w": 0.85, "depth_end": 0.80,
                "pose_w":  1.00, "pose_end":  0.95,
                "tile_w":  0.70, "tile_end":  0.90,
                "ipa_w":   1.15, "ipa_end":   1.00,
            },
        }

        def _do_replicate_generate(
            source_img, tags_str, add_quality,
            ult_pos, ult_neg,
            mode_label, denoise_val, intensity_label,
            resolution_mode,
            steps_, cfg_, sampler_, seed_, width_, height_,
            canny_on, canny_w, canny_end,
            depth_on, depth_w, depth_end,
            pose_on,  pose_w,  pose_end,
            tile_on,  tile_w,  tile_end,
            ipa_on,   ipa_w,   ipa_end,
        ):
            # Apply the intensity preset if one is picked. Sliders/mode reflect
            # whatever the preset wants; explicit user adjustments still get
            # used if they tweak the sliders AFTER picking a preset (Gradio
            # state is just what gets passed in).
            preset = INTENSITY_PRESETS.get(intensity_label)
            if preset:
                mode_label   = preset["mode_label"]
                if preset["denoise"] is not None:
                    denoise_val = preset["denoise"]
                cfg_      = preset["cfg"]
                canny_w   = preset["canny_w"];   canny_end = preset["canny_end"]
                depth_w   = preset["depth_w"];   depth_end = preset["depth_end"]
                pose_w    = preset["pose_w"];    pose_end  = preset["pose_end"]
                tile_w    = preset["tile_w"];    tile_end  = preset["tile_end"]
                ipa_w     = preset["ipa_w"];     ipa_end   = preset["ipa_end"]

            if source_img is None:
                yield None, "⚠️  Upload a reference image first."
                return
            if not tags_str or not tags_str.strip():
                yield None, "⚠️  Click **Analyze image** first."
                return

            # Prefer the Ultimate Prompt if available. The 2nd arg into this
            # handler is the extracted WD14 tag string; the 3rd is the user
            # quality toggle. We swap in the ultimate prompt when it has been
            # populated by the analyze step (via the ultimate_prompt_box).
            # Use Ultimate Prompt if Analyze populated it; otherwise fall back to
            # the raw tag string + standard negative.
            if ult_pos and ult_pos.strip():
                full_pos = ult_pos.strip()
                full_neg = (ult_neg or ILLUSTRIOUS_REPLICATE_NEGATIVE).strip()
            else:
                if add_quality:
                    full_pos = "{}, {}".format(tags_str, ILLUSTRIOUS_REPLICATE_QUALITY)
                    full_neg = ILLUSTRIOUS_REPLICATE_NEGATIVE
                else:
                    full_pos = tags_str
                    full_neg = ""

            # Resolve target resolution FIRST so we can use it as the CN
            # processor_res default. Pre-processing at the actual output bucket
            # (e.g. 1216x832) gives sharper edge/depth maps than the 1024 fallback.
            final_w_status, final_h_status = _compute_output_size(
                resolution_mode, source_img, width_, height_)
            _cn_proc_res = max(int(final_w_status), int(final_h_status), 512)

            # Build CN unit specs
            units_spec = []
            summary = []
            def add(label, cat_pri, module, on, w, end, cat_fallback=None):
                if not on: return
                m = _cn_pick(cat_pri) or (_cn_pick(cat_fallback) if cat_fallback else "")
                if not m:
                    return
                units_spec.append({"module": module, "model": m,
                                   "weight": w, "guidance_end": end,
                                   "processor_res": _cn_proc_res})
                summary.append("{} ({:.2f}, end {:.2f}) - {}".format(label, w, end, m))

            add("Canny",      "canny",     "canny",                 canny_on, canny_w, canny_end, "union")
            add("Depth",      "depth",     "depth_anything_v2",     depth_on, depth_w, depth_end, "union")
            add("OpenPose",   "openpose",  "openpose_full",         pose_on,  pose_w,  pose_end,  "union")
            add("Tile",       "tile",      "tile_resample",         tile_on,  tile_w,  tile_end)
            add("IP-Adapter", "ipadapter", "ip-adapter_clip_sdxl_plus_vith",
                ipa_on, ipa_w, ipa_end)

            if not units_spec:
                yield None, ("⚠️  No matching CN models found. Untick units you "
                             "don't have or install more SDXL CN models.")
                return

            mode_str = "img2img (denoise {:.2f})".format(denoise_val) if mode_label.startswith("img2img") else "txt2img"
            res_str = "{}x{}".format(final_w_status, final_h_status)
            if source_img is not None:
                sw, sh = source_img.size
                res_str += " (source was {}x{})".format(sw, sh)
            yield None, ("Mode: **{}**  ·  intensity: **{}**  ·  output **{}**\n\n"
                         "Building with **{}** ControlNet unit(s):\n\n- {}\n\n"
                         "**Generating...**").format(
                mode_str, intensity_label, res_str,
                len(units_spec), "\n- ".join(summary))

            mode = "img2img" if mode_label.startswith("img2img") else "txt2img"
            # Resolve the final width/height from the resolution mode.
            final_w, final_h = _compute_output_size(
                resolution_mode, source_img, width_, height_)
            img, info = _run_replication_generate(
                source_img, full_pos, full_neg, units_spec,
                steps=steps_, sampler=sampler_, cfg=cfg_,
                width=final_w, height=final_h, seed=seed_,
                mode=mode, denoise=denoise_val,
            )
            if img is None:
                yield None, "❌ **Generation failed**: {}".format(info)
                return

            yield img, ("✅ **Replication done** with {} CN unit(s). Image "
                        "saved to your txt2img output folder."
                        ).format(len(units_spec))

        generate_cn_btn.click(
            fn=_do_replicate_generate,
            inputs=[analyze_image_input, extracted_tags_box, add_quality_cb,
                    ultimate_prompt_box, ultimate_negative_box,
                    replicate_mode_radio, replicate_denoise, replicate_intensity,
                    replicate_resolution_mode,
                    cn_steps, cn_cfg, cn_sampler, cn_seed, cn_width, cn_height,
                    cn_canny_cb,     cn_canny_weight,     cn_canny_end,
                    cn_depth_cb,     cn_depth_weight,     cn_depth_end,
                    cn_pose_cb,      cn_pose_weight,      cn_pose_end,
                    cn_tile_cb,      cn_tile_weight,      cn_tile_end,
                    cn_ipadapter_cb, cn_ipadapter_weight, cn_ipadapter_end],
            outputs=[cn_result_image, cn_status],
        )

        def _format_params(positive: str, negative: str) -> str:
            if not positive and not negative:
                return ""
            lines = [positive or ""]
            if negative:
                lines.append("Negative prompt: {}".format(negative))
            lines.append("Steps: 32, Sampler: Euler a, CFG scale: 6, Size: 1024x1344")
            return "\n".join(lines)

        def _do_build(family_label,
                      subject_l, char_name, custom_subj,
                      pose_l, action_l,
                      shot_l, angle_l,
                      outfit_l,
                      loc_l, time_l, light_l,
                      expr_l, style_l,
                      quality_t, negative_t, year_t, source_t, rating_t,
                      extra_pos, extra_neg,
                      c1, c2, c3, c4, ub,
                      body_l, intimacy_l,
                      artist_l, artist_s, custom_artist,
                      force_female, yuri_on):
            pos, neg, msg = build_prompt(
                family_label, subject_l, char_name, custom_subj,
                pose_l, action_l, shot_l, angle_l, outfit_l,
                loc_l, time_l, light_l, expr_l, style_l,
                quality_t, negative_t, year_t, source_t, rating_t,
                extra_pos, extra_neg,
                char1_traits=c1, char2_traits=c2,
                char3_traits=c3, char4_traits=c4,
                use_break=bool(ub),
                body_label=body_l,
                intimacy_label=intimacy_l,
                artist_label=artist_l,
                artist_strength=float(artist_s) if artist_s is not None else 1.0,
                custom_artist_tag=custom_artist or "",
                force_female_only=bool(force_female),
                yuri=bool(yuri_on),
            )
            params_str = _format_params(pos, neg)
            return pos, neg, msg, params_str

        build_btn.click(
            fn=_do_build,
            inputs=[family,
                    subject_dd, character_name, custom_subject,
                    pose_dd, action_dd,
                    shot_dd, angle_dd,
                    outfit_dd,
                    location_dd, time_dd, lighting_dd,
                    expression_dd, style_dd,
                    quality, negative, year,
                    source, rating,
                    extra_positive, extra_negative,
                    char1_traits, char2_traits, char3_traits, char4_traits,
                    use_break,
                    body_dd, intimacy_dd,
                    artist_dd, artist_strength, custom_artist_tag,
                    force_female_only_cb, yuri_cb],
            outputs=[positive_out, negative_out, status, params_state],
        )

        # ====================================================================
        # Paste-params: wire the three "Send to..." buttons to Forge's paste API.
        # Forge Neo renamed generation_parameters_copypaste -> infotext_utils;
        # try the new name first, fall back to legacy for compatibility.
        # ====================================================================
        _paste_mod = None
        for _mod_name in ("modules.infotext_utils",
                          "modules.generation_parameters_copypaste"):
            try:
                import importlib as _importlib
                _paste_mod = _importlib.import_module(_mod_name)
                break
            except ImportError:
                continue

        if _paste_mod is not None and hasattr(_paste_mod, "register_paste_params_button"):
            # Builder tab -> txt2img / img2img use the assembled params_state
            _paste_mod.register_paste_params_button(
                _paste_mod.ParamBinding(
                    paste_button=send_t2i_btn,
                    tabname="txt2img",
                    source_text_component=params_state,
                )
            )
            _paste_mod.register_paste_params_button(
                _paste_mod.ParamBinding(
                    paste_button=send_i2i_btn,
                    tabname="img2img",
                    source_text_component=params_state,
                )
            )
            # Replicate tab -> txt2img uses the replicate_params_state populated
            # by the image analyzer.
            _paste_mod.register_paste_params_button(
                _paste_mod.ParamBinding(
                    paste_button=replicate_t2i_btn,
                    tabname="txt2img",
                    source_text_component=replicate_params_state,
                    source_image_component=analyze_image_input,
                )
            )

    # ========================================================================
    # Pro Tools tab: token counter, linter, negative presets, prompt presets,
    # wildcards, LoRA suggest, CN recipe presets, auto-CN suggest, batch
    # analyzer, image compare, send-to-ADetailer, recipe card, history,
    # recent artists. Built in a separate Blocks under lib_enhancer/ui_panel.
    # If the import fails we silently skip — the main tab still works.
    # ========================================================================
    pro_entry = None
    try:
        import sys as _sys, os as _os
        _ext_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _ext_root not in _sys.path:
            _sys.path.insert(0, _ext_root)
        from lib_enhancer.ui_panel import build_pro_tools_blocks
        pro_entry = (build_pro_tools_blocks(),
                     "Prompt Enhancer · Pro Tools",
                     "prompt_enhancer_pro")
    except Exception as _e:
        print("[sd-forge-prompt-enhancer] Pro Tools tab unavailable:", _e)

    tabs = [(ui, "Prompt Enhancer", "prompt_enhancer")]
    if pro_entry is not None:
        tabs.append(pro_entry)
    return tabs


script_callbacks.on_ui_tabs(_on_ui_tabs)
