import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.cuda.amp import GradScaler, autocast
import bitsandbytes as bnb
import random

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 8e-5
EPOCHS = 25
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 64
BEAM_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer
MODEL_NAME = "google-t5/t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# Custom dataset
class TagsToDescriptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        input_encoding = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        target_encoding = tokenizer(target_text, max_length=MAX_OUTPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": target_encoding.input_ids.squeeze(),
        }

# Sample data (replace with actual dataset later)
data = [
    (
	"red roses, bouquet, vase, table",
	"a bouquet of red roses in a vase on a table"
    ),
    (
        "golden retriever, park, frisbee, playing",
        "a golden retriever playing with a frisbee in a park",
    ),
    (
	"vintage car, classic, red, shiny",
	"a shiny red classic vintage car"
    ),
    (
        "coffee cup, steam, book, window",
        "a steaming cup of coffee next to an open book by a window",
    ),
    (
        "butterflies, garden, colorful, flowers",
        "colorful butterflies fluttering in a flower garden",
    ),
    (
        "snowman, winter, scarf, carrot nose",
        "a snowman with a scarf and carrot nose in a winter scene",
    ),
    (
        "hot air balloons, sky, colorful, festival",
        "a festival of colorful hot air balloons in the sky",
    ),
    (
        "laptop, coffee shop, working, busy",
        "someone working on a laptop in a busy coffee shop",
    ),
    (
        "waterfall, forest, rocks, mist",
        "a misty waterfall cascading over rocks in a lush forest",
    ),
    (
        "fireworks, night sky, celebration, city",
        "fireworks lighting up the night sky over a city in celebration",
    ),
    (
        "bicycle, old, rusty, leaning on wall",
        "an old rusty bicycle leaning against a wall",
    ),
    (
        "pizza, fresh, steaming, toppings",
        "a fresh, steaming pizza with various toppings",
    ),
    (
        "lighthouse, cliff, stormy sea, waves",
        "a lighthouse on a cliff overlooking a stormy sea with crashing waves",
    ),
    (
        "autumn leaves, red, orange, falling",
        "red and orange autumn leaves falling from trees",
    ),
    (
        "train, railway, countryside, moving",
        "a train moving along a railway through the countryside",
    ),
    (
        "guitar, stage, performer, spotlight",
        "a performer playing guitar on stage under a spotlight",
    ),
    ("graffiti, urban, colorful, wall", "colorful graffiti art on an urban wall"),
    (
        "newborn baby, sleeping, blanket, peaceful",
        "a peaceful newborn baby sleeping wrapped in a blanket",
    ),
    (
        "campfire, night, marshmallows, friends",
        "friends roasting marshmallows over a campfire at night",
    ),
    (
        "sunflower field, summer, yellow, blue sky",
        "a field of yellow sunflowers under a bright blue summer sky",
    ),
    (
        "bookshelf, library, old books, ladder",
        "a tall bookshelf in a library filled with old books and a ladder",
    ),
    (
        "rainy day, umbrella, puddles, city street",
        "people with umbrellas walking on a rainy city street with puddles",
    ),
    (
        "northern lights, aurora, night, stars",
        "the northern lights (aurora) dancing in the night sky full of stars",
    ),
    (
        "sushi, plate, chopsticks, wasabi",
        "a plate of various sushi rolls with chopsticks and wasabi",
    ),
    (
        "hot air balloon, sunrise, landscape, floating",
        "a hot air balloon floating over a landscape at sunrise",
    ),
    (
        "street performer, crowd, music, hat",
        "a street performer playing music for a crowd with a hat for tips",
    ),
    (
        "cherry blossoms, spring, park, pink",
        "pink cherry blossoms blooming in a park during spring",
    ),
    (
        "thunderstorm, lightning, dark clouds, rain",
        "a thunderstorm with lightning striking from dark clouds and heavy rain",
    ),
    (
        "sailboat, ocean, calm water, sunset",
        "a sailboat on calm ocean waters during a beautiful sunset",
    ),
    (
        "old typewriter, desk, paper, vintage",
        "an old vintage typewriter on a desk with a sheet of paper",
    ),
    (
        "ferris wheel, amusement park, night, lights",
        "an illuminated ferris wheel at an amusement park at night",
    ),
    (
        "zen garden, raked sand, rocks, peaceful",
        "a peaceful zen garden with raked sand and carefully placed rocks",
    ),
    (
        "street food, vendor, busy market, steam",
        "a street food vendor selling steaming dishes in a busy market",
    ),
    (
        "ballet dancer, stage, tutu, graceful",
        "a graceful ballet dancer in a tutu performing on stage",
    ),
    (
        "bonfire, beach, friends, night",
        "friends gathered around a bonfire on the beach at night",
    ),
    (
        "misty forest, morning, sunbeams, trees",
        "sunbeams penetrating through a misty forest in the morning",
    ),
    (
        "roller coaster, excitement, speed, amusement park",
        "an exciting high-speed roller coaster at an amusement park",
    ),
    (
        "old vinyl records, player, music notes, retro",
        "old vinyl records and a record player with music notes in a retro setting",
    ),
    (
        "space shuttle, launch, smoke, sky",
        "a space shuttle launching with a plume of smoke against the sky",
    ),
    (
        "windmill, field, blue sky, traditional",
        "a traditional windmill in a field under a clear blue sky",
    ),
    (
        "starry night, silhouette, tree, moon",
        "the silhouette of a tree against a starry night sky with a bright moon",
    ),
    (
        "ice cream cone, melting, summer, colorful",
        "a colorful ice cream cone melting on a hot summer day",
    ),
    (
        "ancient ruins, archaeology, columns, history",
        "ancient ruins with broken columns at an archaeological site",
    ),
    (
        "bonsai tree, pot, zen, miniature",
        "a carefully pruned miniature bonsai tree in a pot, exuding zen",
    ),
    (
        "solar eclipse, sun, moon, astronomical",
        "a solar eclipse with the moon passing in front of the sun",
    ),
    (
        "origami cranes, paper, colorful, art",
        "a collection of colorful paper origami cranes",
    ),
    (
        "lavender field, purple, bees, summer",
        "a vibrant purple lavender field buzzing with bees in summer",
    ),
    (
        "subway, rush hour, commuters, busy",
        "a busy subway station during rush hour filled with commuters",
    ),
    (
        "sand dunes, desert, camel, sunset",
        "sand dunes in a desert with a camel silhouette at sunset",
    ),
    (
        "fog, bridge, mysterious, morning",
        "a mysterious fog-covered bridge in the early morning",
    ),
    (
        "coral reef, tropical fish, underwater, colorful",
        "a colorful coral reef teeming with tropical fish underwater",
    ),
    (
        "radio telescope, night sky, stars, science",
        "a large radio telescope pointing at the starry night sky",
    ),
    (
        "venetian mask, carnival, feathers, ornate",
        "an ornate Venetian mask with feathers for a carnival",
    ),
    (
        "shipwreck, underwater, fish, coral",
        "an old shipwreck underwater surrounded by fish and coral",
    ),
    (
        "stone arch, desert, rock formation, erosion",
        "a natural stone arch rock formation in the desert, shaped by erosion",
    ),
    (
        "autumn forest, fog, path, colorful leaves",
        "a foggy path through an autumn forest with colorful leaves",
    ),
    (
        "solar panels, field, renewable energy, sun",
        "a field filled with solar panels harnessing renewable energy from the sun",
    ),
    (
        "street art, mural, urban, large scale",
        "a large-scale mural of street art on an urban building",
    ),
    (
        "tidal wave, surfer, ocean, extreme sport",
        "a surfer riding a massive tidal wave in the ocean",
    ),
    (
        "christmas market, festive, lights, snow",
        "a festive Christmas market with twinkling lights and snow",
    ),
    (
        "hummingbird, flower, nectar, hovering",
        "a hummingbird hovering while drinking nectar from a flower",
    ),
    (
        "volcano eruption, lava, smoke, night",
        "a volcano erupting at night with flowing lava and billowing smoke",
    ),
    (
        "rustic barn, countryside, weathered wood, field",
        "a weathered wooden rustic barn in a countryside field",
    ),
    (
        "medieval castle, stone walls, towers, moat",
        "a medieval castle with stone walls, towers, and a surrounding moat",
    ),
    (
        "koi pond, japanese garden, water lilies, peaceful",
        "a peaceful koi pond with water lilies in a Japanese garden",
    ),
    (
        "subway graffiti, urban art, train, colorful",
        "a subway train covered in colorful graffiti urban art",
    ),
    (
        "wine cellar, barrels, bottles, dim lighting",
        "a dimly lit wine cellar filled with wooden barrels and bottles",
    ),
    (
        "hanging gardens, lush, waterfall, exotic",
        "lush hanging gardens with a cascading waterfall and exotic plants",
    ),
    (
        "frozen bubble, winter, soap, crystallized",
        "a crystallized frozen bubble in a winter landscape",
    ),
    (
        "pottery workshop, clay, wheel, artisan",
        "an artisan working with clay on a wheel in a pottery workshop",
    ),
    (
        "fire breather, performance, flame, night",
        "a fire breather performing with a large flame at night",
    ),
    (
        "rice terraces, green, agriculture, asia",
        "green stepped rice terraces in an Asian agricultural landscape",
    ),
    (
        "abandoned amusement park, rusty, overgrown, eerie",
        "an eerie abandoned amusement park with rusty rides overgrown with plants",
    ),
    (
        "sand mandala, colorful, intricate, tibetan",
        "an intricate and colorful Tibetan sand mandala",
    ),
    (
        "aurora borealis, arctic, reflection, water",
        "the aurora borealis reflected in still arctic waters",
    ),
    (
        "venetian canal, gondola, bridges, old buildings",
        "a Venetian canal with a gondola passing under bridges between old buildings",
    ),
    (
        "crop circles, aerial view, field, mysterious",
        "mysterious crop circles viewed from an aerial perspective in a field",
    ),
    (
        "paper lanterns, festival, night sky, glowing",
        "glowing paper lanterns floating in the night sky during a festival",
    ),
    (
        "redwood forest, tall trees, sunlight, mist",
        "sunlight filtering through tall trees in a misty redwood forest",
    ),
    (
        "great wall of china, ancient, winding, mountains",
        "the Great Wall of China winding through mountains",
    ),
    (
        "monastery, mountains, tibet, peaceful",
        "a peaceful Tibetan monastery perched in the mountains",
    ),
    (
        "street musicians, acoustic guitar, crowd, city",
        "street musicians playing acoustic guitar for a crowd in the city",
    ),
    (
        "antique clock, gears, time, close-up",
        "a close-up view of the intricate gears inside an antique clock",
    ),
    (
        "reef shark, underwater, coral, blue",
        "a reef shark swimming in blue waters near a coral reef",
    ),
    (
        "stonehenge, ancient, mystical, sunset",
        "the ancient and mystical Stonehenge silhouetted at sunset",
    ),
    (
        "alpine meadow, wildflowers, mountains, summer",
        "a summer alpine meadow filled with wildflowers and mountain views",
    ),
    (
        "urban rooftop garden, city view, green, sustainable",
        "a sustainable urban rooftop garden with a city skyline view",
    ),
    (
        "milky way, night sky, silhouette, mountains",
        "the Milky Way stretching across the night sky over silhouetted mountains",
    ),
    (
        "ice hotel, snow, sculpture, winter",
        "an ice hotel with intricate snow sculptures in a winter wonderland",
    ),
    (
        "thermal springs, steam, nature, relaxation",
        "steaming thermal springs in a natural setting for relaxation",
    ),
    (
        "floating market, boats, colorful, busy",
        "a busy floating market with colorful boats selling various goods",
    ),
    (
        "fairy tale cottage, thatched roof, garden, quaint",
        "a quaint fairy tale cottage with a thatched roof and charming garden",
    ),
    (
        "bioluminescent waves, beach, night, glowing",
        "glowing bioluminescent waves washing up on a beach at night",
    ),
    (
        "ancient tree, twisted, gnarly, old",
        "an ancient, gnarly tree with twisted branches showing its age",
    ),
    (
        "moroccan tiles, intricate, colorful, pattern",
        "intricate and colorful patterns of traditional Moroccan tiles",
    ),
    (
        "astronaut, space walk, earth view, stars",
        "an astronaut on a space walk with a view of Earth and stars",
    ),
    (
        "frozen waterfall, winter, icicles, blue",
        "a frozen waterfall with blue icicles in a winter landscape",
    ),
    (
        "himalayan mountains, snow peaks, majestic, sunrise",
        "majestic snow-covered Himalayan mountain peaks at sunrise",
    ),
    (
        "street performer, levitation, illusion, crowd",
        "a street performer creating a levitation illusion for an amazed crowd",
    ),
    (
        "beach, sunset, ocean, silhouette",
        "a silhouette of people walking on a beach at sunset",
    ),
    (
        "mountain, snow, peak, climber",
        "a climber reaching the snow-capped peak of a mountain",
    ),
    (
        "cat, sleeping, sunbeam, windowsill",
        "a cat sleeping in a sunbeam on a windowsill",
    ),
    (
        "city skyline, night, lights, reflection",
        "a city skyline at night with lights reflecting on water",
    ),
    (
        "child, playground, swing, laughing",
        "a laughing child swinging high on a playground swing",
    ),
    (
        "campfire, marshmallows, friends, camping",
        "friends roasting marshmallows over a campfire while camping",
    ),
    (
        "rainbow, after rain, vibrant, sky",
        "a vibrant rainbow arching across the sky after rain",
    ),
    (
        "old bookstore, shelves, dusty, cozy",
        "a cozy old bookstore with dusty shelves full of books",
    ),
    (
        "chef, kitchen, cooking, steam",
        "a chef cooking in a steamy professional kitchen",
    ),
    (
        "ballet dancer, performance, graceful, stage",
        "a graceful ballet dancer performing on stage",
    ),
    (
        "street musician, guitar, crowd, hat",
        "a street musician playing guitar with a crowd around and a hat for tips",
    ),
    (
        "rainy day, umbrella, puddles, reflection",
        "people with umbrellas reflected in puddles on a rainy day",
    ),
    (
        "sports car, racetrack, speed, blur",
        "a sports car speeding around a racetrack, creating a motion blur",
    ),
    (
        "zen garden, rake, stones, peaceful",
        "a peaceful zen garden with raked sand and carefully placed stones",
    ),
    (
        "fruit market, colorful, fresh, busy",
        "a busy fruit market with colorful displays of fresh produce",
    ),
    (
        "telescope, observatory, stars, night",
        "a large telescope in an observatory pointed at a starry night sky",
    ),
    (
        "paint brushes, palette, canvas, artist",
        "an artist's palette with paint brushes next to a blank canvas",
    ),
    ("surfer, wave, beach, action", "a surfer riding a large wave near a beach"),
    (
        "castle, medieval, stone, tower",
        "a medieval stone castle with tall towers and a moat",
    ),
    (
        "bonsai tree, small, artistic, pot",
        "a small, artistically pruned bonsai tree in a ceramic pot",
    ),
    (
        "street food, vendor, steam, crowd",
        "a street food vendor serving steaming dishes to a hungry crowd",
    ),
    (
        "drone, aerial view, landscape, flying",
        "an aerial view of a landscape captured by a flying drone",
    ),
    (
        "northern lights, aurora, sky, night",
        "the northern lights (aurora) dancing in the night sky",
    ),
    (
        "vineyard, grapes, sunset, rolling hills",
        "a vineyard with ripe grapes on rolling hills at sunset",
    ),
    (
        "origami, paper, crane, delicate",
        "a delicate paper crane made with origami technique",
    ),
    (
        "wedding, bride, groom, ceremony",
        "a bride and groom during their wedding ceremony",
    ),
    (
        "coral reef, fish, colorful, underwater",
        "a colorful coral reef teeming with various fish underwater",
    ),
    (
        "roller coaster, amusement park, thrill, speed",
        "a thrilling roller coaster speeding through an amusement park",
    ),
    (
        "taj mahal, reflection, architecture, marble",
        "the Taj Mahal and its reflection in water, showcasing its marble architecture",
    ),
    (
        "windmill, field, wind, blades",
        "a windmill with spinning blades in a field on a windy day",
    ),
    (
        "bonfire, beach, night, friends",
        "friends gathered around a bonfire on the beach at night",
    ),
    ("farm, barn, tractor, fields", "a red barn and tractor in fields on a farm"),
    (
        "sushi, plate, chopsticks, wasabi",
        "a plate of various sushi rolls with chopsticks and wasabi",
    ),
    (
        "ice skating, rink, winter, pirouette",
        "a figure skater performing a pirouette on an ice skating rink",
    ),
    (
        "hot spring, steam, rocks, relaxation",
        "people relaxing in a steaming natural hot spring surrounded by rocks",
    ),
    (
        "space shuttle, launch, smoke, sky",
        "a space shuttle launching with a plume of smoke against the sky",
    ),
    (
        "desert, cactus, sand dunes, heat waves",
        "a lone cactus in a desert with sand dunes and visible heat waves",
    ),
    (
        "gondola, venice, canal, romantic",
        "a romantic gondola ride through a canal in Venice",
    ),
    (
        "great wall, china, ancient, winding",
        "the ancient Great Wall of China winding over mountains",
    ),
    (
        "tulip fields, colorful, rows, netherlands",
        "rows of colorful tulip fields stretching to the horizon in the Netherlands",
    ),
    (
        "street art, mural, urban, large",
        "a large, colorful mural of street art on an urban building",
    ),
    (
        "deer, forest, antlers, majestic",
        "a majestic deer with large antlers standing in a forest",
    ),
    (
        "solar eclipse, sun, moon, dramatic",
        "a dramatic solar eclipse with the moon covering the sun",
    ),
    (
        "lavender field, purple, fragrant, bees",
        "a fragrant purple lavender field buzzing with bees",
    ),
    (
        "shipwreck, underwater, coral, fish",
        "an old shipwreck underwater, covered in coral and surrounded by fish",
    ),
    (
        "marathon, runners, city, crowd",
        "a crowd of runners participating in a city marathon",
    ),
    (
        "cherry blossoms, spring, pink, delicate",
        "delicate pink cherry blossoms blooming in spring",
    ),
    (
        "haunted house, spooky, old, Halloween",
        "a spooky old haunted house decorated for Halloween",
    ),
    (
        "kayak, river rapids, adventure, splash",
        "an adventurer in a kayak splashing through river rapids",
    ),
    (
        "rock climbing, cliff, ropes, challenging",
        "a challenging rock climbing route on a steep cliff with ropes",
    ),
    (
        "sand castle, beach, detailed, towers",
        "a detailed sand castle with multiple towers on a beach",
    ),
    (
        "wine cellar, barrels, bottles, dim light",
        "a dim wine cellar filled with wooden barrels and bottles",
    ),
    (
        "redwood forest, tall trees, sunbeams, mist",
        "sunbeams filtering through mist in a forest of tall redwood trees",
    ),
    (
        "lego, colorful, bricks, creation",
        "a colorful creation made from various Lego bricks",
    ),
    (
        "skyscraper, modern, glass, tall",
        "a modern, tall glass skyscraper reaching into the sky",
    ),
    (
        "pottery wheel, clay, hands, shaping",
        "hands shaping clay on a spinning pottery wheel",
    ),
    ("concert, crowd, stage, lights", "a large crowd at a concert with a lit-up stage"),
    (
        "bubble, iridescent, floating, delicate",
        "a delicate, iridescent bubble floating in the air",
    ),
    (
        "aquarium, fish, coral, viewing tunnel",
        "people in a viewing tunnel of a large aquarium surrounded by fish and coral",
    ),
    (
        "farmers market, fresh produce, stalls, busy",
        "a busy farmers market with stalls full of fresh produce",
    ),
    (
        "ice sculpture, intricate, frozen, art",
        "an intricate ice sculpture glittering in the light",
    ),
    (
        "kite flying, beach, wind, colorful",
        "colorful kites flying high on a windy beach",
    ),
    (
        "subway, busy, commuters, underground",
        "busy commuters in an underground subway station",
    ),
    (
        "zen monk, meditation, temple, peaceful",
        "a peaceful zen monk meditating in a temple",
    ),
    (
        "safari, animals, savanna, jeep",
        "a jeep on safari observing animals in the savanna",
    ),
    (
        "carnival, ferris wheel, lights, night",
        "a carnival at night with a brightly lit ferris wheel",
    ),
    ("beehive, bees, honey, busy", "a busy beehive with bees making honey"),
    ("dandelion, seeds, blowing, wind", "dandelion seeds blowing in the wind"),
    (
        "astronaut, space walk, earth, stars",
        "an astronaut on a space walk with Earth and stars in the background",
    ),
    (
        "terraced rice fields, green, asia, steps",
        "green terraced rice fields stepping down a hillside in Asia",
    ),
    (
        "venetian mask, ornate, feathers, gold",
        "an ornate Venetian mask decorated with feathers and gold",
    ),
    (
        "bodypaint, colorful, artist, model",
        "a model covered in colorful bodypaint by an artist",
    ),
    (
        "morning dew, grass, droplets, sunrise",
        "droplets of morning dew on grass blades at sunrise",
    ),
    (
        "time lapse, city, day to night, busy",
        "a time lapse of a busy city transitioning from day to night",
    ),
    (
        "archaeological dig, ancient, artifacts, excavation",
        "an archaeological excavation uncovering ancient artifacts",
    ),
    (
        "snow globe, winter scene, shaking, glitter",
        "a winter scene inside a snow globe being shaken, with glitter swirling",
    ),
    (
        "volcano, eruption, lava, smoke",
        "a volcano erupting with flowing lava and billowing smoke",
    ),
    (
        "jazz band, saxophone, piano, performance",
        "a jazz band performing with a saxophonist and pianist",
    ),
    (
        "paper lanterns, festival, glowing, night",
        "glowing paper lanterns floating in the night sky during a festival",
    ),
    (
        "mosaic, colorful tiles, pattern, art",
        "a colorful mosaic artwork made from tiny tiles in an intricate pattern",
    ),
    (
        "thunderstorm, lightning, dark clouds, rain",
        "a thunderstorm with lightning striking from dark clouds and heavy rain",
    ),
    (
        "spiral staircase, architecture, winding, perspective",
        "a winding spiral staircase viewed from a unique perspective",
    ),
    (
        "easter eggs, decorated, colorful, basket",
        "a basket filled with colorfully decorated Easter eggs",
    ),
    (
        "hourglass, sand, time, flowing",
        "sand flowing through an hourglass, marking the passage of time",
    ),
    (
        "parkour, urban, jumping, skill",
        "a skilled parkour artist jumping between urban structures",
    ),
    (
        "jellyfish, bioluminescent, dark, ocean",
        "bioluminescent jellyfish glowing in a dark ocean",
    ),
    (
        "autumn forest, colors, leaves, path",
        "a path through an autumn forest with colorful fallen leaves",
    ),
    (
        "sandstorm, desert, wind, dramatic",
        "a dramatic sandstorm sweeping across a desert landscape",
    ),
    (
        "confetti, celebration, falling, colorful",
        "colorful confetti falling during a celebration",
    ),
    (
        "moonlight, silhouette, tree, night",
        "the silhouette of a tree against a bright full moon at night",
    ),
    (
        "tide pool, sea life, rocky, coast",
        "a rocky coastal tide pool filled with diverse sea life",
    ),
    (
        "mirrored building, reflection, sky, modern",
        "a modern mirrored building reflecting the sky and surroundings",
    ),
    (
        "bioluminescent plankton, beach, night, glowing",
        "glowing bioluminescent plankton on a beach at night",
    ),
    (
        "frozen bubble, winter, crystal, delicate",
        "a delicate frozen bubble with crystal patterns in winter",
    ),
    (
        "damascus steel, knife, pattern, craftsmanship",
        "a knife made of Damascus steel showcasing intricate pattern and craftsmanship",
    ),
    (
        "hummingbird, flower, hovering, tiny",
        "a tiny hummingbird hovering near a flower",
    ),
    (
        "subway, graffiti, urban, colorful",
        "a subway car covered in colorful urban graffiti",
    ),
    (
        "glacial lake, turquoise, mountains, reflection",
        "a turquoise glacial lake reflecting surrounding mountains",
    ),
    (
        "beach, sunset, silhouette, palm trees, ocean, waves, sand, couple, walking, romantic, seagulls, orange sky, reflection, footprints, tropical",
        "A romantic couple walking hand in hand along a tropical beach at sunset, their silhouettes framed by palm trees against an orange sky, with seagulls flying overhead and footprints trailing behind them in the wet sand",
    ),
    (
        "mountain, snow-capped, hiking, backpack, trail, forest, pine trees, clear sky, stream, rocks, adventure, outdoor gear, binoculars, map, compass",
        "An adventurous hiker with a large backpack navigating a challenging trail through a pine forest towards a snow-capped mountain peak, consulting a map and compass while crossing a rocky stream",
    ),
    (
        "city skyline, night, skyscrapers, lights, reflection, river, bridge, traffic, long exposure, urban, modern, architecture, illuminated, busy, metropolis",
        "A bustling metropolis at night with illuminated skyscrapers reflecting in a river, a bridge filled with streaks of car lights from long exposure, showcasing the vibrant urban life and modern architecture",
    ),
    (
        "farmer's market, fresh produce, colorful, vegetables, fruits, organic, local, vendors, baskets, crowds, street, umbrellas, signs, cash, reusable bags",
        "A lively farmer's market street scene with colorful stalls overflowing with fresh, organic produce, busy vendors, and crowds of shoppers with reusable bags, all under a canopy of bright umbrellas and hand-painted signs",
    ),
    (
        "space, stars, galaxy, nebula, planets, astronaut, floating, spaceship, Earth, cosmic, exploration, science fiction, weightlessness, darkness, wonder",
        "An astronaut floating in the vastness of space, surrounded by a dazzling array of stars, colorful nebulae, and distant planets, with Earth visible in the background and a futuristic spaceship nearby",
    ),
    (
        "underwater, coral reef, tropical fish, scuba diver, sea turtle, bubbles, blue, clear water, sunbeams, colorful, marine life, exploration, camera, fins, mask",
        "A scuba diver with a camera swimming through a vibrant coral reef teeming with tropical fish and sea turtles, sunbeams piercing the clear blue water as bubbles rise around the explorer",
    ),
    (
        "vineyard, wine, grapes, sunset, rolling hills, barrels, winery, tasting, glasses, cheese platter, romantic, countryside, rustic, harvest, vines",
        "A romantic sunset over rolling hills covered in lush grapevines, with a rustic winery in the distance and a foreground of wine barrels, glasses, and a gourmet cheese platter set for tasting",
    ),
    (
        "circus, big top, acrobats, clowns, trapeze, audience, ringmaster, animals, colorful, performance, lights, costumes, excitement, tent, applause",
        "The interior of a grand circus big top during a spectacular performance, with acrobats on trapezes, clowns entertaining the crowd, and a ringmaster directing the show, all under bright lights and colorful decorations",
    ),
    (
        "winter, snow, cabin, forest, smoke, chimney, cozy, warm lights, evergreens, frozen lake, mountains, starry sky, silence, isolation, peace",
        "A cozy log cabin nestled in a snowy forest, smoke curling from its chimney, warm lights glowing from within, overlooking a frozen lake and surrounded by towering evergreens under a starry winter sky",
    ),
    (
        "safari, savanna, elephants, giraffes, jeep, tourists, binoculars, cameras, sunset, acacia trees, dust, adventure, wildlife, nature, expedition",
        "An exciting safari expedition with tourists in a jeep observing a herd of elephants and giraffes on the savanna at sunset, dust kicked up around the vehicle, with cameras and binoculars capturing the moment",
    ),
    (
        "amusement park, roller coaster, ferris wheel, cotton candy, crowds, families, excitement, lights, carousel, games, tickets, screams, laughter, night, fun",
        "A bustling amusement park at night, filled with excited families, featuring a towering roller coaster, glowing ferris wheel, and colorful carousel, the air filled with the scent of cotton candy, screams of delight, and laughter",
    ),
    (
        "studio, artist, painting, canvas, brushes, palette, easel, creativity, inspiration, mess, colors, concentration, natural light, artwork, passion",
        "An artist deeply focused in their sunlit studio, surrounded by canvases, paintbrushes, and a messy palette, working passionately on a large, colorful painting propped on an easel",
    ),
    (
        "storm, lightning, dark clouds, rain, wind, trees bending, dramatic, power, nature, danger, excitement, weather, flash, thunder, atmosphere",
        "A dramatic storm scene with forked lightning illuminating dark, roiling clouds, rain lashing down, and trees bending in the powerful wind, capturing the raw power and danger of nature",
    ),
    (
        "library, books, shelves, reading, quiet, study, knowledge, literature, cozy chairs, lamps, students, concentration, wooden tables, old building, wisdom",
        "The interior of a grand old library with towering shelves filled with books, students quietly studying at wooden tables under soft lamplight, and cozy reading nooks tucked between rows of literary treasures",
    ),
    (
        "festival, music, crowd, stage, performers, lights, dancing, energy, excitement, night, outdoor, food stalls, costumes, joy, celebration",
        "A lively outdoor music festival at night with a huge crowd dancing in front of a brightly lit stage, performers energizing the audience, surrounded by food stalls and people in colorful festival attire",
    ),
    (
        "garden, flowers, butterflies, bees, sunshine, colors, tranquil, bench, path, fountain, blooming, nature, peaceful, spring, beauty",
        "A tranquil garden scene on a sunny spring day, with a winding path leading to a bench near a fountain, surrounded by blooming flowers of every color, with butterflies and bees flitting from blossom to blossom",
    ),
    (
        "airport, planes, travelers, luggage, departures board, security, gates, international, busy, excitement, reunion, goodbyes, tickets, passports, announcements",
        "A bustling international airport terminal with planes visible through large windows, travelers rushing with luggage, emotional reunions and goodbyes, and a large departures board displaying flights from around the world",
    ),
    (
        "construction site, cranes, workers, scaffolding, hard hats, blueprints, trucks, building, progress, safety gear, busy, teamwork, urban development, machines, steel",
        "A busy urban construction site with towering cranes, workers in hard hats and safety gear consulting blueprints, trucks delivering materials, and the steel skeleton of a new building rising amidst scaffolding",
    ),
    (
        "sushi restaurant, chef, knife skills, fresh fish, rice, seaweed, wasabi, soy sauce, chopsticks, counter, customers, Japanese, culinary art, precision, delicacy",
        "A skilled sushi chef behind a counter, demonstrating precise knife skills as he prepares fresh fish, with customers watching in anticipation, surrounded by the artful presentation of sushi rolls, wasabi, and soy sauce",
    ),
    (
        "medieval castle, stone walls, towers, flags, moat, drawbridge, knights, armor, throne room, tapestries, torches, history, fortress, royal, ancient",
        "An imposing medieval castle with high stone walls and towers flying colorful flags, protected by a moat and drawbridge, with knights in shining armor guarding a richly decorated throne room inside",
    ),
    (
        "yoga class, meditation, mats, instructor, poses, zen, relaxation, fitness, mindfulness, stretching, peaceful, studio, balance, breathe, harmony",
        "A serene yoga studio filled with people on colorful mats, following an instructor through various poses, focusing on breath and mindfulness in a peaceful atmosphere of relaxation and harmony",
    ),
    (
        "rock concert, band, guitars, drums, crowd surfing, mosh pit, stage diving, loud, energetic, fans, merchandise, backstage, amplifiers, tour bus, rock and roll",
        "An electrifying rock concert with a band ripping through guitar solos and pounding drums, fans crowd surfing and moshing, while others buy merchandise, all culminating in an unforgettable night of loud, energetic rock and roll",
    ),
    (
        "wedding, bride, groom, ceremony, flowers, dress, suit, rings, vows, guests, celebration, love, romance, photographer, cake",
        "A beautiful outdoor wedding ceremony with a radiant bride in a flowing white dress and a handsome groom in a sharp suit, exchanging vows and rings surrounded by flowers, emotional guests, and a photographer capturing every moment",
    ),
    (
        "laboratory, scientist, microscope, test tubes, experiments, research, data, computer, lab coat, safety goggles, chemicals, discovery, innovation, technology, science",
        "A cutting-edge laboratory with scientists in white coats and safety goggles conducting experiments, peering through microscopes, analyzing data on computers, and handling chemicals, all in pursuit of scientific discovery and innovation",
    ),
    (
        "street food, night market, vendors, steam, aroma, crowds, neon lights, chopsticks, woks, spices, local cuisine, hustle and bustle, flavors, culture, delicious",
        "A vibrant night market filled with street food vendors, steam rising from sizzling woks, the air thick with the aroma of spices and local cuisine, as crowds navigate the hustle and bustle under the glow of neon lights",
    ),
    (
        "ballet performance, dancers, tutus, pointe shoes, grace, theater, audience, orchestra pit, stage lights, choreography, elegance, flexibility, music, art, emotion",
        "A breathtaking ballet performance on a grand theater stage, dancers in tutus and pointe shoes moving with incredible grace and flexibility, the orchestra playing emotive music as the audience watches in awe under dramatic stage lighting",
    ),
    (
        "shipwreck, underwater, coral, fish, scuba divers, exploration, treasure, mystery, marine life, ocean floor, algae, rusty, history, adventure, discovery",
        "Scuba divers exploring a mysterious shipwreck on the ocean floor, now covered in coral and algae, surrounded by colorful fish and marine life, searching for hidden treasures and piecing together the vessel's history",
    ),
    (
        "cat cafe, kittens, coffee, relaxation, play area, cat toys, customers, cozy, furry friends, adoption, animal lovers, comfort, unique, interaction, happiness",
        "A cozy cat cafe filled with playful kittens and contented cats, customers sipping coffee while interacting with their furry friends, surrounded by cat toys and climbing structures in a unique, relaxing environment",
    ),
    (
        "street art, mural, spray paint, urban, culture, creativity, wall, artist, social commentary, vibrant colors, street scene, passersby, transformation, expression, public space",
        "A large, vibrant mural being created by a street artist using spray paint, transforming a dull urban wall into a powerful piece of social commentary, as passersby stop to watch the creative process unfold",
    ),
    (
        "1girl,solo,blonde hair,twintails,school uniform,smiling,waving,classroom",
        "a blonde girl with twintails in a school uniform, smiling and waving in a classroom",
    ),
    (
        "2girls,brown hair,short hair,long hair,hugging,laughing,park bench",
        "two girls with brown hair, one with short hair and one with long hair, hugging and laughing on a park bench",
    ),
    (
        "1boy,spiky hair,glasses,reading book,library,concentrating",
        "a boy with spiky hair and glasses, concentrating while reading a book in a library",
    ),
    (
        "3girls,diverse hair colors,pajamas,pillow fight,bedroom,excited",
        "three girls with different hair colors having an excited pillow fight in pajamas in a bedroom",
    ),
    (
        "1girl,1boy,holding hands,blushing,school hallway,nervous",
        "a girl and a boy holding hands and blushing nervously in a school hallway",
    ),
    (
        "1girl,solo,red hair,freckles,painting,art studio,focused",
        "a red-haired girl with freckles, focused on painting in an art studio",
    ),
    (
        "2boys,playing video games,competitive,living room,excited",
        "two boys playing video games competitively in a living room, looking excited",
    ),
    (
        "1girl,twin braids,overalls,gardening,backyard,happy",
        "a girl with twin braids wearing overalls, happily gardening in a backyard",
    ),
    (
        "1girl,solo,blue hair,long hair,brown eyes,standing,sandals,black sandals,looking at viewer,peace hand sign,smile,shoes,black shoes,socks,long socks",
        "a girl with long blue hair and brown eyes, standing and smiling, making a peace hand sign, wearing black sandals and long socks",
    ),
    (
        "2girls,pink hair,short hair,t-shirt,red t-shirt,blue eyes,pants,black pants,matching outfit,1boy,long hair,angry,crying,outdoors",
        "two girls with short pink hair and blue eyes wearing matching red t-shirts and black pants, with a long-haired boy who looks angry and is crying, all outdoors",
    ),
    (
        "1girl,blonde hair,medium hair,green eyes,dress,floral dress,sitting,chair,wooden chair,reading,book,glasses,round glasses,smile",
        "a girl with medium-length blonde hair and green eyes, sitting on a wooden chair, wearing a floral dress and round glasses, smiling while reading a book",
    ),
    (
        "2boys,1girl,school uniform,backpack,walking,sidewalk,autumn,falling leaves,laughing",
        "two boys and one girl in school uniforms with backpacks, walking on a sidewalk and laughing among falling autumn leaves",
    ),
    (
        "1boy,curly hair,red hair,freckles,hoodie,blue hoodie,jeans,torn jeans,sneakers,white sneakers,skateboard,park",
        "a boy with curly red hair and freckles, wearing a blue hoodie, torn jeans, and white sneakers, holding a skateboard in a park",
    ),
    (
        "3girls,ponytail,braids,short hair,sundress,yellow sundress,white dress,blue dress,picnic,blanket,grass,eating,sandwiches",
        "three girls at a picnic: one with a ponytail in a yellow sundress, one with braids in a white dress, and one with short hair in a blue dress, all sitting on a blanket on grass eating sandwiches",
    ),
    (
        "1boy,2girls,siblings,couch,watching TV,popcorn,bowl,pajamas,striped pajamas,polka dot pajamas,plain pajamas",
        "a boy and two girls, likely siblings, sitting on a couch watching TV in pajamas: one in striped, one in polka dot, and one in plain pajamas, sharing a bowl of popcorn",
    ),
    (
        "2boys,twins,identical twins,soccer uniform,red uniform,cleats,black cleats,ball,soccer ball,field,grass,competitive,facing each other",
        "two identical twin boys in red soccer uniforms and black cleats, facing each other competitively on a grassy field with a soccer ball between them",
    ),
    (
        "1girl,hijab,white hijab,long dress,green dress,cooking,kitchen,stirring,pot,stove,apron,floral apron,smile",
        "a girl wearing a white hijab and long green dress with a floral apron, smiling while stirring a pot on a stove in a kitchen",
    ),
    (
        "3boys,baseball caps,backwards cap,t-shirts,shorts,knee-length shorts,fishing,pier,water,lake,fishing rods,tackle box",
        "three boys wearing baseball caps (one backwards) and t-shirts with knee-length shorts, fishing from a pier on a lake, holding fishing rods with a tackle box nearby",
    ),
    (
        "2girls,1boy,formal wear,prom,dress,long dress,tuxedo,black tuxedo,corsage,boutonniere,posing,photos,garden",
        "two girls in long dresses and a boy in a black tuxedo, dressed for prom with corsages and a boutonniere, posing for photos in a garden",
    ),
    (
        "1boy,glasses,round glasses,sweater vest,bow tie,khaki pants,loafers,brown loafers,textbook,classroom,chalkboard,teaching",
        "a boy with round glasses wearing a sweater vest, bow tie, khaki pants, and brown loafers, teaching in front of a chalkboard in a classroom, holding a textbook",
    ),
    (
        "2girls,best friends,ice cream,cone,waffle cone,chocolate ice cream,vanilla ice cream,bench,park,summer,sunglasses,tank top,shorts,denim shorts",
        "two best friends sitting on a park bench in summer, wearing tank tops, denim shorts, and sunglasses, eating ice cream: one chocolate in a waffle cone, one vanilla in a regular cone",
    ),
    (
        "1girl,1boy,couple,holding hands,walking,beach,sunset,sundress,flowy dress,button-up shirt,rolled sleeves,bare feet,footprints,sand",
        "a couple holding hands and walking barefoot on a beach at sunset, leaving footprints in the sand: the girl in a flowy sundress, the boy in a button-up shirt with rolled sleeves",
    ),
    (
        "3girls,sleepover,pillow fight,pajamas,sleeping bags,popcorn,movie night,living room,laughter,pillows,flying feathers",
        "three girls having a sleepover in a living room, engaged in a pillow fight with feathers flying, wearing pajamas with sleeping bags and popcorn nearby, laughing during movie night",
    ),
    (
        "1boy,2girls,science fair,project,poster board,lab coats,safety goggles,beakers,colorful liquids,proud,explaining,judges",
        "a boy and two girls at a science fair, wearing lab coats and safety goggles, proudly explaining their project with a poster board and beakers of colorful liquids to judges",
    ),
    (
        "2boys,tree house,rope ladder,binoculars,telescope,map,adventure,backpack,camping gear,forest,tall trees",
        "two boys in a tree house in a forest of tall trees, one climbing a rope ladder, the other looking through binoculars, with a telescope, map, backpack, and camping gear visible",
    ),
    (
        "1girl,ballet,tutu,pink tutu,leotard,ballet shoes,pointe shoes,bun,hair bun,mirror,dance studio,barre,stretching",
        "a girl with her hair in a bun, wearing a pink tutu, leotard, and pointe shoes, stretching at a barre in front of a mirror in a dance studio",
    ),
    (
        "2boys,1girl,band practice,garage,drums,electric guitar,microphone,amplifier,sheet music,focused,jamming",
        "two boys and a girl having band practice in a garage: one on drums, one on electric guitar, and one at a microphone, with an amplifier and sheet music visible, all looking focused while jamming",
    ),
    (
        "1girl,artist,painting,easel,canvas,palette,brushes,smock,messy hair,concentrated,colorful,abstract art",
        "a girl with messy hair wearing a smock, concentrating while painting an abstract, colorful piece on a canvas on an easel, holding a palette and brushes",
    ),
    (
        "3boys,video game tournament,gaming chairs,headsets,controllers,large screen,excited,competitive,energy drinks",
        "three boys participating in a video game tournament, sitting in gaming chairs with headsets and controllers, looking excited and competitive in front of a large screen, with energy drinks nearby",
    ),
    (
        "2girls,cooking class,aprons,chef hats,mixing bowls,ingredients,flour,eggs,measuring cups,kitchen,teamwork",
        "two girls in a cooking class wearing aprons and chef hats, working together in a kitchen with mixing bowls, flour, eggs, and measuring cups, displaying teamwork",
    ),
    (
        "1boy,1girl,debate competition,podiums,microphones,formal attire,suit,blazer,skirt,notes,audience,intense",
        "a boy in a suit and a girl in a blazer and skirt at podiums with microphones, participating in an intense debate competition with notes in hand, facing an audience",
    ),
    (
        "3girls,cheerleading,uniforms,pom poms,high kicks,pyramid formation,gym,school spirit,energetic",
        "three cheerleaders in uniforms with pom poms, performing high kicks and a pyramid formation in a gym, displaying energetic school spirit",
    ),
    (
        "2boys,fort building,blankets,chairs,pillows,flashlights,living room,imaginative play,teamwork",
        "two boys building a fort in a living room using blankets, chairs, and pillows, holding flashlights and engaging in imaginative play with teamwork",
    ),
    (
        "1girl,violin practice,music stand,sheet music,bow,concentrated,living room,carpet,standing,focused",
        "a girl standing on a carpet in a living room, concentrated and focused while practicing violin, with a music stand, sheet music, and bow visible",
    ),
    (
        "2boys,1girl,science experiment,lab,test tubes,safety goggles,lab coats,notebook,taking notes,curious,excited",
        "two boys and a girl in a lab conducting a science experiment, wearing safety goggles and lab coats, with test tubes and a notebook for taking notes, looking curious and excited",
    ),
    (
        "1boy,2girls,homework,study group,textbooks,laptops,notebooks,living room,coffee table,snacks,focused",
        "a boy and two girls in a study group doing homework in a living room, gathered around a coffee table with textbooks, laptops, notebooks, and snacks, all looking focused",
    ),
    (
        "3boys,skateboard park,ramps,helmets,knee pads,elbow pads,tricks,cheering,encouraging,urban setting",
        "three boys at a skateboard park in an urban setting, wearing helmets, knee pads, and elbow pads, performing tricks on ramps while cheering and encouraging each other",
    ),
    (
        "2girls,1boy,school play,costumes,stage,curtain,props,makeup,nervous,excited,backstage",
        "two girls and a boy backstage at a school play, wearing costumes and makeup, surrounded by props and a curtain, looking both nervous and excited",
    ),
    (
        "1girl,horseback riding,helmet,riding boots,jodhpurs,horse,stables,brushing,grooming,affectionate",
        "a girl wearing a helmet, riding boots, and jodhpurs, affectionately brushing and grooming a horse near stables",
    ),
    (
        "2boys,fishing trip,fishing rods,tackle box,life jackets,small boat,lake,early morning,fog,peaceful",
        "two boys on an early morning fishing trip, sitting in a small boat on a foggy lake, wearing life jackets and holding fishing rods, with a tackle box nearby, looking peaceful",
    ),
    (
        "3girls,slumber party,sleeping bags,popcorn,nail polish,magazines,pillows,giggling,whispering,bedroom",
        "three girls having a slumber party in a bedroom, lying in sleeping bags with popcorn, nail polish, magazines, and pillows scattered around, giggling and whispering",
    ),
    (
        "1boy,2girls,talent show,microphone,guitar,keyboard,stage,spotlights,audience,nervous excitement",
        "a boy and two girls on stage at a talent show under spotlights, one at a microphone, one with a guitar, and one at a keyboard, facing an audience with nervous excitement",
    ),
    (
        "2boys,1girl,snowball fight,winter,snow,mittens,scarves,hats,jackets,laughing,playful",
        "two boys and a girl having a snowball fight in winter, wearing mittens, scarves, hats, and jackets, laughing and being playful in the snow",
    ),
    (
        "3boys,treehouse construction,tools,wood,nails,rope,ladder,teamwork,planning,backyard",
        "three boys working together to construct a treehouse in a backyard, using tools, wood, nails, and rope, with a ladder nearby, showing teamwork and planning",
    ),
    (
        "2girls,baking cookies,kitchen,mixer,cookie cutters,flour,aprons,oven mitts,decorating,teamwork",
        "two girls in a kitchen wearing aprons and oven mitts, baking cookies together with a mixer, cookie cutters, and flour, decorating and showing teamwork",
    ),
    (
        "1boy,1girl,tennis match,tennis rackets,tennis court,net,tennis balls,sporty outfits,competitive,focused",
        "a boy and a girl playing a competitive tennis match on a court, holding tennis rackets and focused on the ball, wearing sporty outfits",
    ),
    (
        "3girls,jump rope,playground,recess,school uniforms,sneakers,pigtails,braids,ponytail,laughing",
        "three girls at a playground during recess, jumping rope while wearing school uniforms and sneakers, their hair in pigtails, braids, and a ponytail, all laughing",
    ),
    (
        "2boys,1girl,lemonade stand,table,pitcher,cups,sign,money box,sidewalk,summer day,entrepreneurial",
        "two boys and a girl running a lemonade stand on a sidewalk on a summer day, with a table, pitcher, cups, sign, and money box, looking entrepreneurial",
    ),
    (
        "1girl,2boys,card game,deck of cards,table,chairs,competitive,focused,living room,after school",
        "a girl and two boys playing a competitive card game after school, sitting at a table with chairs in a living room, focused on their hands and the deck of cards",
    ),
    (
        "3boys,video game marathon,console,controllers,snacks,soda,couch,blankets,excited,late night",
        "three boys having a late-night video game marathon, sitting on a couch with blankets, holding controllers, surrounded by snacks and soda, looking excited",
    ),
    (
        "2girls,1boy,school project,poster board,markers,glue,scissors,books,collaborative,focused,classroom",
        "two girls and a boy working collaboratively on a school project in a classroom, using a poster board, markers, glue, scissors, and books, all looking focused",
    ),
    (
        "1girl,2boys,obstacle course,backyard,tires,ropes,cones,stopwatch,cheering,competitive,summer",
        "a girl and two boys running a backyard obstacle course in summer, with tires, ropes, and cones, one holding a stopwatch, others cheering, all looking competitive",
    ),
    (
        "3girls,makeover party,makeup,nail polish,hair accessories,mirror,laughing,bedroom,sleepover,fun",
        "three girls having a makeover party at a sleepover in a bedroom, using makeup, nail polish, and hair accessories, laughing in front of a mirror and having fun",
    ),
    (
        "2boys,1girl,science club,microscope,petri dishes,lab coats,safety goggles,notebook,curious,excited",
        "two boys and a girl in a science club, wearing lab coats and safety goggles, examining petri dishes under a microscope, taking notes in a notebook, looking curious and excited",
    ),
    (
        "1boy,2girls,impromptu dance party,living room,music player,colorful lights,dance moves,laughter,joy",
        "a boy and two girls having an impromptu dance party in a living room, with a music player and colorful lights, showing off dance moves, filled with laughter and joy",
    ),
    (
        "3boys,building blocks,tower,concentration,teamwork,carpet,living room,creative play",
        "three boys on a carpet in a living room, concentrating on building a tall tower with building blocks, showing teamwork and engaging in creative play",
    ),
    (
        "2girls,1boy,painting class,easels,canvases,palettes,brushes,aprons,focused,artistic,studio",
        "two girls and a boy in a painting class, standing at easels with canvases, holding palettes and brushes, wearing aprons, looking focused and artistic in a studio setting",
    ),
    (
        "1girl,2boys,board game night,dice,game pieces,cards,snacks,table,competitive,fun,family room",
        "a girl and two boys having a board game night in a family room, rolling dice and moving game pieces, with cards and snacks on the table, looking competitive and having fun",
    ),
    (
        "3girls,jump rope competition,playground,cheering,judges,stopwatch,determined,athletic,school yard",
        "three girls in a jump rope competition on a school playground, one jumping while others cheer, with judges holding a stopwatch, all looking determined and athletic",
    ),
    (
        "2boys,1girl,robotics club,robot,computer,tools,wires,collaborative,focused,workshop",
        "two boys and a girl in a robotics club, working collaboratively on a robot in a workshop, using a computer and tools, surrounded by wires, all looking focused",
    ),
    (
        "1boy,2girls,talent show rehearsal,microphone,guitar,keyboard,stage,practicing,supportive,auditorium",
        "a boy and two girls rehearsing for a talent show on an auditorium stage, one singing into a microphone, one playing guitar, and one at a keyboard, all practicing and being supportive",
    ),
    (
        "3boys,capture the flag,park,teams,flags,running,strategy,competitive,outdoors,summer",
        "three boys playing capture the flag in a park during summer, running with flags, showing strategy and competitiveness in an outdoor setting",
    ),
    (
        "2girls,1boy,book club meeting,different books,discussion,cozy corner,bean bags,snacks,engaged,library",
        "two girls and a boy at a book club meeting in a cozy library corner, sitting on bean bags with different books, engaged in discussion",
    ),
    (
        "2boys,short hair,brown hair,green eyes,twins,matching outfits,striped shirts,blue jeans,sneakers,white sneakers,playing,soccer,backyard",
        "twin boys with short brown hair and green eyes, wearing matching striped shirts, blue jeans, and white sneakers, playing soccer in a backyard",
    ),
    (
        "1boy,curly hair,red hair,hazel eyes,glasses,round glasses,freckles,sweater,green sweater,khaki pants,loafers,brown loafers,studying,library",
        "a boy with curly red hair, hazel eyes, freckles, and round glasses, wearing a green sweater, khaki pants, and brown loafers, studying in a library",
    ),
    (
        "3girls,different heights,braids,ponytail,short hair,brunette,blonde,redhead,school uniforms,backpacks,walking,sidewalk,laughing",
        "three girls of different heights walking on a sidewalk and laughing: one tall brunette with braids, one medium-height blonde with a ponytail, and one short redhead with short hair, all in school uniforms with backpacks",
    ),
    (
        "1girl,2boys,siblings,mixed race,curly hair,straight hair,brown skin,pale skin,t-shirts,shorts,barefoot,playing,board game,living room",
        "three mixed-race siblings playing a board game in the living room: a girl with curly hair and brown skin, one boy with straight hair and pale skin, and another with curly hair and brown skin, all wearing t-shirts and shorts, barefoot",
    ),
    (
        "2girls,long hair,short hair,blue eyes,brown eyes,best friends,crop top,denim shorts,sneakers,taking selfie,mall,shopping bags",
        "two best friends taking a selfie at the mall with shopping bags: one tall girl with long hair and blue eyes wearing a crop top, the other shorter with short hair and brown eyes in denim shorts, both wearing sneakers",
    ),
    (
        "1boy,spiky hair,blonde hair,green eyes,athletic build,basketball jersey,shorts,high tops,dribbling,basketball,court",
        "an athletic boy with spiky blonde hair and green eyes, wearing a basketball jersey, shorts, and high tops, dribbling a basketball on a court",
    ),
    (
        "3boys,different ages,brothers,buzz cut,bowl cut,long hair,brown hair,black hair,blonde hair,pajamas,couch,video games,competitive",
        "three brothers of different ages playing video games competitively on a couch: the oldest with a buzz cut and brown hair, the middle with a bowl cut and black hair, the youngest with long blonde hair, all in pajamas",
    ),
    (
        "1girl,pixie cut,purple hair,nose piercing,tattoo,arm tattoo,tank top,black tank top,ripped jeans,combat boots,guitar,practice,bedroom",
        "a girl with a purple pixie cut, nose piercing, and arm tattoo, wearing a black tank top, ripped jeans, and combat boots, practicing guitar in her bedroom",
    ),
    (
        "2boys,1girl,triplets,identical,long hair,ponytails,freckles,green eyes,overalls,striped shirts,gardening,backyard,watering cans",
        "triplets with long hair in ponytails, freckles, and green eyes, wearing overalls over striped shirts, gardening in the backyard with watering cans",
    ),
    (
        "1boy,afro,dark skin,brown eyes,tall,lanky,jersey,basketball shorts,sneakers,shooting hoops,playground",
        "a tall, lanky boy with an afro, dark skin, and brown eyes, wearing a jersey and basketball shorts with sneakers, shooting hoops on a playground",
    ),
    (
        "2girls,pigtails,braids,glasses,braces,school uniforms,backpacks,lunchboxes,waiting,bus stop,morning",
        "two girls waiting at a bus stop in the morning: one with pigtails and glasses, the other with braids and braces, both in school uniforms with backpacks and lunchboxes",
    ),
    (
        "1girl,2boys,cousins,different heights,chubby,skinny,average build,t-shirts,jeans,shorts,sandals,picnic,park",
        "three cousins having a picnic in the park: a chubby girl of average height, a tall skinny boy, and a short average-build boy, all wearing t-shirts, a mix of jeans and shorts, and sandals",
    ),
    (
        "3girls,cheerleaders,high ponytails,makeup,uniforms,pom poms,practicing,gym,pyramid formation",
        "three cheerleaders with high ponytails and makeup, wearing uniforms and holding pom poms, practicing a pyramid formation in a gym",
    ),
    (
        "1boy,mohawk,earrings,leather jacket,band t-shirt,ripped jeans,boots,playing,electric guitar,garage",
        "a boy with a mohawk and earrings, wearing a leather jacket over a band t-shirt, ripped jeans, and boots, playing an electric guitar in a garage",
    ),
    (
        "2boys,twins,crew cut,freckles,buck teeth,polo shirts,khaki shorts,boat shoes,fishing,pier,lake",
        "twin boys with crew cuts, freckles, and buck teeth, wearing polo shirts, khaki shorts, and boat shoes, fishing from a pier on a lake",
    ),
    (
        "1girl,long hair,ombre hair,thick eyebrows,nose ring,crop top,high waisted jeans,platform shoes,taking photos,camera,city street",
        "a girl with long ombre hair, thick eyebrows, and a nose ring, wearing a crop top, high-waisted jeans, and platform shoes, taking photos with a camera on a city street",
    ),
    (
        "3boys,different ethnicities,varied hairstyles,hoodies,jeans,sneakers,skateboards,skate park,tricks",
        "three boys of different ethnicities with varied hairstyles, all wearing hoodies, jeans, and sneakers, performing tricks on skateboards at a skate park",
    ),
    (
        "2girls,1boy,drama club,costumes,makeup,wigs,scripts,stage,rehearsing,school auditorium",
        "two girls and a boy from drama club rehearsing on a school auditorium stage, wearing costumes, makeup, and wigs, holding scripts",
    ),
    (
        "1girl,hijab,modest dress,glasses,book bag,textbooks,walking,university campus,determined",
        "a girl wearing a hijab and modest dress with glasses, carrying a book bag and textbooks, walking determinedly across a university campus",
    ),
    (
        "2boys,surfer hair,tanned skin,board shorts,rash guards,surfboards,beach,waves,excited",
        "two boys with surfer hair and tanned skin, wearing board shorts and rash guards, excitedly carrying surfboards towards waves on a beach",
    ),
    (
        "3girls,ballerinas,buns,leotards,tights,tutus,ballet shoes,stretching,barre,mirror,dance studio",
        "three ballerinas with hair in buns, wearing leotards, tights, tutus, and ballet shoes, stretching at a barre in front of a mirror in a dance studio",
    ),
    (
        "1boy,2girls,science fair,lab coats,safety goggles,project display,volcano model,nervous,excited,school gym",
        "a boy and two girls at a science fair in a school gym, wearing lab coats and safety goggles, standing nervously but excitedly by their volcano model project display",
    ),
    (
        "2boys,1girl,band,singer,guitarist,drummer,garage,instruments,practicing,focused",
        "a three-piece band practicing in a garage: a girl singer, a boy guitarist, and a boy drummer, all focused on their instruments",
    ),
    (
        "1girl,2boys,pokemon cosplay,costumes,wigs,props,posing,comic convention,excited",
        "a girl and two boys in Pokemon cosplay with detailed costumes, wigs, and props, posing excitedly at a comic convention",
    ),
    (
        "3girls,sleepover,pajamas,face masks,nail polish,sleeping bags,pillow fight,laughter,bedroom",
        "three girls having a sleepover in a bedroom, wearing pajamas and face masks, surrounded by sleeping bags and nail polish, engaged in a pillow fight and laughing",
    ),
    (
        "1boy,glasses,braces,bow tie,sweater vest,khakis,loafers,spelling bee,stage,microphone,nervous",
        "a boy with glasses and braces, wearing a bow tie, sweater vest, khakis, and loafers, standing nervously on a stage with a microphone at a spelling bee",
    ),
    (
        "2girls,1boy,siblings,mixed race,curly hair,straight hair,family portrait,matching outfits,studio,smiling",
        "three mixed-race siblings posing for a family portrait in a studio: two girls with curly hair and a boy with straight hair, all in matching outfits and smiling",
    ),
    (
        "3boys,team jersey,shorts,knee pads,sports glasses,focused,volleyball,gym,game",
        "three boys wearing team jerseys, shorts, knee pads, and sports glasses, focused during a volleyball game in a gym",
    ),
    (
        "1girl,2boys,art class,aprons,easels,paintbrushes,palettes,canvases,concentrated,classroom",
        "a girl and two boys in an art class, wearing aprons and standing at easels with paintbrushes and palettes, concentrating on their canvases in a classroom",
    ),
    (
        "2girls,1boy,debate team,formal wear,notecards,podium,auditorium,serious,confident",
        "two girls and a boy on a debate team, dressed in formal wear, holding notecards at a podium in an auditorium, looking serious and confident",
    ),
    (
        "3girls,different heights,yoga class,yoga mats,leggings,tank tops,ponytails,stretching,gym",
        "three girls of different heights in a yoga class at a gym, wearing leggings and tank tops with ponytails, stretching on yoga mats",
    ),
    (
        "1boy,2girls,cooking class,aprons,chef hats,mixing bowls,ingredients,kitchen,teamwork",
        "a boy and two girls in a cooking class, wearing aprons and chef hats, working together with mixing bowls and ingredients in a kitchen",
    ),
    (
        "2boys,1girl,robotics club,safety goggles,gloves,tools,robot parts,workbench,focused,workshop",
        "two boys and a girl in a robotics club, wearing safety goggles and gloves, focused on assembling robot parts with tools at a workbench in a workshop",
    ),
    (
        "3boys,different ages,brothers,video games,controllers,couch,snacks,competitive,living room",
        "three brothers of different ages playing video games competitively in a living room, sitting on a couch with controllers and surrounded by snacks",
    ),
    (
        "1girl,2boys,school play,costumes,makeup,props,backstage,nervous,excited,mirror",
        "a girl and two boys preparing for a school play backstage, in costumes and makeup, holding props, looking both nervous and excited in front of a mirror",
    ),
    (
        "2girls,1boy,marching band,uniforms,instruments,practice,field,focused,sweating",
        "two girls and a boy in marching band uniforms, holding instruments and practicing on a field, looking focused and sweating",
    ),
    (
        "3girls,sports team,jerseys,shorts,cleats,soccer ball,field,stretching,pre-game",
        "three girls on a sports team, wearing jerseys, shorts, and cleats, stretching with a soccer ball on a field before a game",
    ),
    (
        "1boy,2girls,talent show,microphone,guitar,keyboard,stage,spotlights,performing,audience",
        "a boy and two girls performing at a talent show on a stage under spotlights, one with a microphone, one with a guitar, and one at a keyboard, facing an audience",
    ),
    (
        "2boys,1girl,nature hike,backpacks,water bottles,hiking boots,map,forest trail,exploring",
        "two boys and a girl on a nature hike, wearing backpacks and hiking boots, carrying water bottles and a map, exploring a forest trail",
    ),
    (
        "3boys,skateboard park,helmets,pads,skateboards,ramps,tricks,cheering,urban",
        "three boys at a skateboard park wearing helmets and pads, performing tricks on ramps with their skateboards, cheering each other on in an urban setting",
    ),
    (
        "1girl,2boys,chess club,chess boards,concentration,tournament,classroom,silent,focused",
        "a girl and two boys in a chess club, intensely focused and silent during a tournament in a classroom, concentrating on chess boards",
    ),
    (
        "2girls,1boy,school newspaper,notebooks,cameras,press passes,interviewing,school hallway",
        "two girls and a boy working for the school newspaper, holding notebooks and cameras with press passes, interviewing someone in a school hallway",
    ),
    (
        "3girls,slumber party,sleeping bags,popcorn,movies,pillows,giggling,whispering,living room",
        "three girls having a slumber party in a living room, in sleeping bags with popcorn and pillows, giggling and whispering while watching movies",
    ),
    (
        "1boy,2girls,science experiment,lab coats,safety goggles,test tubes,beakers,lab,excited",
        "a boy and two girls conducting a science experiment in a lab, wearing lab coats and safety goggles, excitedly handling test tubes and beakers",
    ),
    (
        "2boys,1girl,trivia team,buzzers,team shirts,concentration,auditorium,competition",
        "two boys and a girl on a trivia team, wearing team shirts and poised over buzzers, concentrating during a competition in an auditorium",
    ),
    (
        "3boys,car wash fundraiser,wet,soapy,sponges,hoses,buckets,laughing,parking lot",
        "three boys at a car wash fundraiser in a parking lot, wet and soapy, laughing while holding sponges, hoses, and buckets",
    ),
    (
        "1girl,2boys,community service,trash bags,gloves,picking up litter,park,teamwork",
        "a girl and two boys doing community service in a park, wearing gloves and holding trash bags, working as a team to pick up litter",
    ),
    (
        "2girls,1boy,yearbook committee,cameras,laptops,layout designs,discussion,classroom",
        "two girls and a boy on the yearbook committee in a classroom, using cameras and laptops, discussing layout designs",
    ),
    (
        "3girls,different heights,dance class,leotards,tights,ballet shoes,barre,mirror,practicing",
        "three girls of different heights in a dance class, wearing leotards, tights, and ballet shoes, practicing at a barre in front of a mirror",
    ),
    (
        "2boys,identical twins,buzz cut,blonde hair,freckles,tall,lanky,tank tops,board shorts,flip flops,beach volleyball,sand",
        "identical twin boys with blonde buzz cuts and freckles, both tall and lanky, wearing tank tops, board shorts, and flip flops, playing beach volleyball in the sand",
    ),
    (
        "3girls,diverse,afro,straight hair,hijab,different heights,sundresses,sandals,picnic,blanket,park",
        "three diverse girls having a picnic on a blanket in a park: one tall with an afro, one medium height with straight hair, and one shorter wearing a hijab, all in sundresses and sandals",
    ),
    (
        "1boy,dreadlocks,dark skin,muscular,sleeve tattoo,tank top,joggers,running shoes,basketball,court",
        "a muscular boy with dreadlocks and dark skin, sporting a sleeve tattoo, wearing a tank top, joggers, and running shoes, playing basketball on a court",
    ),
    (
        "2girls,1boy,siblings,redheads,pale skin,freckles,matching outfits,striped shirts,jeans,converse shoes,family photo,backyard",
        "three redheaded siblings with pale skin and freckles in a family photo in the backyard: two girls and one boy wearing matching striped shirts, jeans, and Converse shoes",
    ),
    (
        "1girl,long braids,glasses,braces,overalls,striped shirt,sneakers,painting,easel,art studio",
        "a girl with long braids, glasses, and braces, wearing overalls over a striped shirt and sneakers, painting at an easel in an art studio",
    ),
    (
        "2boys,mohawks,punk style,leather jackets,band t-shirts,ripped jeans,combat boots,electric guitars,garage band,practice",
        "two boys with mohawks and punk style, wearing leather jackets over band t-shirts, ripped jeans, and combat boots, practicing with electric guitars in a garage band",
    ),
    (
        "3boys,different ages,bowl cut,spiky hair,long hair,polo shirts,khaki shorts,loafers,golfing,course",
        "three boys of different ages golfing on a course: one with a bowl cut, one with spiky hair, and one with long hair, all wearing polo shirts, khaki shorts, and loafers",
    ),
    (
        "1girl,2boys,debate team,formal wear,blazers,dress shirts,slacks,dress,heels,podium,auditorium",
        "a debate team in an auditorium: a girl in a dress and heels, and two boys in blazers, dress shirts, and slacks, standing at a podium",
    ),
    (
        "2girls,1boy,mixed race,curly hair,straight hair,lab coats,safety goggles,experiment,science fair,school gym",
        "three mixed-race students at a science fair in a school gym: two girls with curly hair and a boy with straight hair, all wearing lab coats and safety goggles, conducting an experiment",
    ),
    (
        "1boy,afro,thick glasses,bow tie,suspenders,high-waisted pants,dress shoes,spelling bee,stage,microphone",
        "a boy with an afro and thick glasses, wearing a bow tie, suspenders, high-waisted pants, and dress shoes, competing in a spelling bee on stage with a microphone",
    ),
    (
        "3girls,cheerleaders,high ponytails,uniforms,pom poms,sneakers,pyramid formation,gym floor",
        "three cheerleaders with high ponytails in uniforms and sneakers, holding pom poms and forming a pyramid on a gym floor",
    ),
    (
        "2boys,1girl,band members,colored hair,piercings,ripped jeans,band t-shirts,boots,instruments,garage",
        "three band members in a garage: two boys and a girl with colored hair and piercings, wearing ripped jeans, band t-shirts, and boots, holding various instruments",
    ),
    (
        "1girl,hijab,modest dress,glasses,backpack,textbooks,walking,university campus,autumn",
        "a girl wearing a hijab, modest dress, and glasses, carrying a backpack and textbooks while walking across a university campus in autumn",
    ),
    (
        "2boys,surfer look,long hair,tan skin,board shorts,tank tops,barefoot,surfboards,beach,waves",
        "two boys with surfer looks: long hair and tan skin, wearing board shorts and tank tops, barefoot on a beach with surfboards near waves",
    ),
    (
        "3girls,ballet dancers,hair buns,leotards,tights,tutus,pointe shoes,stretching,barre,mirror",
        "three ballet dancers with hair in buns, wearing leotards, tights, tutus, and pointe shoes, stretching at a barre in front of a mirror",
    ),
    (
        "1boy,2girls,theater kids,costumes,makeup,wigs,scripts,stage,rehearsal,school auditorium",
        "three theater kids rehearsing on a school auditorium stage: a boy and two girls in costumes, makeup, and wigs, holding scripts",
    ),
    (
        "2boys,1girl,robotics team,safety glasses,gloves,team shirts,jeans,sneakers,robot,competition arena",
        "a robotics team at a competition arena: two boys and a girl wearing safety glasses, gloves, team shirts, jeans, and sneakers, working on a robot",
    ),
    (
        "3boys,soccer team,jerseys,shorts,cleats,knee socks,different hairstyles,pre-game,field",
        "three boys from a soccer team on a field pre-game, wearing jerseys, shorts, cleats, and knee socks, each with a different hairstyle",
    ),
    (
        "1girl,2boys,art class,aprons,berets,easels,palettes,paintbrushes,canvases,messy,studio",
        "an art class in a studio: a girl and two boys wearing aprons and berets, standing at easels with palettes and paintbrushes, working on canvases, looking messy",
    ),
    (
        "2girls,1boy,marching band,uniforms,hats,instruments,practice,football field,hot day",
        "two girls and a boy in marching band uniforms with hats, holding instruments while practicing on a football field on a hot day",
    ),
    (
        "3girls,swim team,swimsuits,swim caps,goggles,towels,pool deck,pre-race,nervous",
        "three girls from a swim team on a pool deck pre-race looking nervous, wearing swimsuits, swim caps, and goggles, holding towels",
    ),
    (
        "1boy,2girls,coding club,laptops,glasses,hoodies,jeans,focused,computer lab",
        "a coding club in a computer lab: a boy and two girls wearing glasses and hoodies with jeans, focused on their laptops",
    ),
    (
        "2boys,1girl,debate tournament,suits,blazer,skirt,dress shoes,notecards,water bottles,auditorium",
        "a debate tournament in an auditorium: two boys in suits and a girl in a blazer and skirt, all wearing dress shoes and holding notecards and water bottles",
    ),
    (
        "3boys,skateboarders,helmets,pads,graphic tees,cargo shorts,skate shoes,skateboards,skate park",
        "three skateboarders at a skate park: boys wearing helmets, pads, graphic tees, cargo shorts, and skate shoes, with their skateboards",
    ),
    (
        "1girl,long hair,blue eyes,green dress,smiling",
        "a girl with long hair and blue eyes, wearing a green dress, smiling",
    ),
    (
        "2boys,red shirts,jeans,playing soccer,park",
        "two boys wearing red shirts and jeans, playing soccer in a park",
    ),
    (
        "3girls,blonde hair,black hair,brunette,cheerleading,high school",
        "three girls cheerleading at high school; one has blonde hair, one has black hair, and one is a brunette",
    ),
    (
        "1boy,freckles,glasses,reading,library,focused",
        "a boy with freckles and glasses, reading a book in a library, looking focused",
    ),
    (
        "2girls,twins,ponytails,purple dresses,dancing,excited",
        "twin girls with ponytails, wearing purple dresses, dancing excitedly",
    ),
    (
        "1girl,short,amelia taylor,red hair,blue eyes,pink dress,happy",
        "a short girl named amelia taylor with red hair and blue eyes, wearing a pink dress, looking happy",
    ),
    (
        "2boys,brown hair,blue shirts,playing basketball,court,competitive",
        "two boys with brown hair, wearing blue shirts, playing basketball on a court, looking competitive",
    ),
    (
        "1girl,long curly hair,green eyes,overalls,gardening,backyard",
        "a girl with long curly hair and green eyes, wearing overalls, gardening in a backyard",
    ),
    (
        "3boys,black hair,green shirts,watching movie,living room,relaxed",
        "three boys with black hair, wearing green shirts, watching a movie in the living room, looking relaxed",
    ),
    (
        "1boy,short hair,blue eyes,yellow t-shirt,skateboarding,park,energetic",
        "a boy with short hair and blue eyes, wearing a yellow t-shirt, skateboarding in a park, looking energetic",
    ),
    (
        "1girl,tall,isabella stone,brown hair,blue eyes,purple dress,smiling",
        "a tall girl named isabella stone with brown hair and blue eyes, wearing a purple dress, smiling",
    ),
    (
        "2girls,red hair,blue hair,pink skirts,playing hopscotch,sidewalk",
        "two girls playing hopscotch on a sidewalk; one has red hair and the other has blue hair, both wearing pink skirts",
    ),
    (
        "1boy,curly hair,green eyes,orange hoodie,reading book,library,focused",
        "a boy with curly hair and green eyes, wearing an orange hoodie, reading a book in a library, looking focused",
    ),
    (
        "2boys,twin brothers,blonde hair,green eyes,red shorts,blue t-shirts,beach,building sandcastle",
        "twin brothers with blonde hair and green eyes, wearing red shorts and blue t-shirts, building a sandcastle on a beach",
    ),
    (
        "1girl,long hair,black eyes,green t-shirt,playing violin,bedroom,concentrated",
        "a girl with long hair and black eyes, wearing a green t-shirt, playing the violin in her bedroom, looking concentrated",
    ),
    (
        "1boy,short,samuel johnson,brown hair,glasses,white shirt,playing piano,living room",
        "a short boy named samuel johnson with brown hair and glasses, wearing a white shirt, playing the piano in the living room",
    ),
    (
        "3girls,black hair,brown eyes,blue skirts,white blouses,singing,karaoke machine,living room",
        "three girls with black hair and brown eyes, wearing blue skirts and white blouses, singing with a karaoke machine in the living room",
    ),
    (
        "1girl,ponytail,green eyes,red dress,dancing,ballroom,joyful",
        "a girl with a ponytail and green eyes, wearing a red dress, dancing in a ballroom, looking joyful",
    ),
    (
        "2boys,blonde hair,blue eyes,white shirts,jeans,fishing,river,happy",
        "two boys with blonde hair and blue eyes, wearing white shirts and jeans, fishing by a river, looking happy",
    ),
    (
        "1girl,long curly hair,blue eyes,yellow dress,smiling,field of flowers",
        "a girl with long curly hair and blue eyes, wearing a yellow dress, smiling in a field of flowers",
    ),
    (
        "1girl,medium height,olivia martin,red hair,green eyes,blue skirt,purple blouse,running,park,excited",
        "a medium height girl named olivia martin with red hair and green eyes, wearing a blue skirt and a purple blouse, running in a park, looking excited",
    ),
    (
        "2girls,brown hair,green eyes,pink shirts,yellow pants,jumping rope,backyard,energetic",
        "two girls with brown hair and green eyes, wearing pink shirts and yellow pants, jumping rope in a backyard, looking energetic",
    ),
    (
        "1boy,freckles,blue eyes,green t-shirt,playing guitar,bedroom,focused",
        "a boy with freckles and blue eyes, wearing a green t-shirt, playing the guitar in his bedroom, looking focused",
    ),
    (
        "3boys,black hair,brown eyes,blue jeans,red t-shirts,playing video games,living room,excited",
        "three boys with black hair and brown eyes, wearing blue jeans and red t-shirts, playing video games in the living room, looking excited",
    ),
    (
        "1girl,long hair,blue eyes,green dress,sketch,smiling",
        "a girl with long hair and blue eyes, wearing a green dress, smiling, in the style of a sketch",
    ),
    (
        "1boy,tall,james smith,curly hair,green eyes,blue hoodie,black pants,reading comic book,library,engrossed",
        "a tall boy named james smith with curly hair and green eyes, wearing a blue hoodie and black pants, reading a comic book in the library, looking engrossed",
    ),
    (
        "2boys,black hair,brown eyes,green shirts,blue shorts,playing chess,park,concentrated",
        "two boys with black hair and brown eyes, wearing green shirts and blue shorts, playing chess in a park, looking concentrated",
    ),
    (
        "1girl,short hair,green eyes,purple dress,painting,canvas,focused",
        "a girl with short hair and green eyes, wearing a purple dress, painting on a canvas, looking focused",
    ),
    (
        "2girls,red hair,blue hair,orange dresses,playing with dolls,bedroom,happy",
        "two girls playing with dolls in a bedroom; one has red hair and the other has blue hair, both wearing orange dresses, looking happy",
    ),
    (
        "3boys,blonde hair,blue eyes,white shirts,black pants,practicing soccer,field,determined",
        "three boys with blonde hair and blue eyes, wearing white shirts and black pants, practicing soccer on a field, looking determined",
    ),
    (
        "1girl,medium height,sophia williams,black hair,brown eyes,yellow dress,playing flute,orchestra,focused",
        "a medium height girl named sophia williams with black hair and brown eyes, wearing a yellow dress, playing the flute in an orchestra, looking focused",
    ),
    (
        "2boys,curly hair,blue eyes,red shirts,black shorts,playing basketball,court,competitive",
        "two boys with curly hair and blue eyes, wearing red shirts and black shorts, playing basketball on a court, looking competitive",
    ),
    (
        "1girl,long curly hair,green eyes,blue dress,playing violin,bedroom,concentrated",
        "a girl with long curly hair and green eyes, wearing a blue dress, playing the violin in her bedroom, looking concentrated",
    ),
    (
        "2girls,twin sisters,blonde hair,brown eyes,purple dresses,playing hopscotch,sidewalk,happy",
        "twin sisters with blonde hair and brown eyes, wearing purple dresses, playing hopscotch on a sidewalk, looking happy",
    ),
    (
        "1boy,short hair,black eyes,yellow t-shirt,skateboarding,park,energetic",
        "a boy with short hair and black eyes, wearing a yellow t-shirt, skateboarding in a park, looking energetic",
    ),
    (
        "1girl,tall,emma johnson,brown hair,green eyes,pink dress,smiling",
        "a tall girl named emma johnson with brown hair and green eyes, wearing a pink dress, smiling",
    ),
    (
        "2boys,black hair,blue eyes,white shirts,jeans,fishing,lake,happy",
        "two boys with black hair and blue eyes, wearing white shirts and jeans, fishing at a lake, looking happy",
    ),
    (
        "1girl,long hair,blue eyes,green t-shirt,playing piano,living room,focused",
        "a girl with long hair and blue eyes, wearing a green t-shirt, playing the piano in the living room, looking focused",
    ),
    (
        "3girls,red hair,black hair,brunette,blue dresses,dancing,party,joyful",
        "three girls dancing at a party; one has red hair, one has black hair, and one is a brunette, all wearing blue dresses, looking joyful",
    ),
    (
        "1boy,freckles,glasses,green hoodie,reading book,library,engrossed",
        "a boy with freckles and glasses, wearing a green hoodie, reading a book in a library, looking engrossed",
    ),
    (
        "1girl,medium height,layla davis,red hair,brown eyes,blue skirt,purple blouse,running,park,excited",
        "a medium height girl named layla davis with red hair and brown eyes, wearing a blue skirt and a purple blouse, running in a park, looking excited",
    ),
    (
        "2boys,red shirt,blue eyes, green eyes,3d,full body,sport,soccer",
        "Full body 3d render of two boys playing soccer. One boy with a red shirt and with blue eyes, and another boy with green eyes.",
    ),
    (
        "1girl,twin braids, amelia specie,overalls,black overalls,gardening,backyard,happy,sketch,portrait",
        "Sketch of a portrait of a girl named amelia specie with twin braids wearing black overalls, happily gardening in a backyard",
    ),
    (
        "1girl,cowboy shot,long hair, green hair, long sleeves, dress shirt,vest,brown vest,skirt,pleated skirt,arisa mifuku,outdoors,smiling",
        "Outdoor cowboy shot of a smiling girl named arisa mifuku with green hair and wearing a dress shirt with long sleeves, covered by a vest, as well as wearing a pleated skirt.",
    ),
    (
        "1boy,short, jason mavery,blonde hair,green eyes,t-shirt,blue t-shirt,shorts,red shorts, beach, digital art,portrait",
        "Digital art portrait of a short boy named jason mavery with blonde hair and green eyes, wearing a blue t-shirt and red shorts on a beach.",
    ),
    (
        "2girls, red hair, blue hair,1girl,black hair, 3d, full body,park,smiling",
        "Full body 3d render of three girls in a park. One girl with red hair, one girl with blue hair, and one girl with black hair, all smiling.",
    ),
    (
        "1girl, long hair, brown hair, red dress,portrait, sketch,happy, zoe wildly",
        "Sketch portrait of a happy girl named zoe wildly with long brown hair wearing a red dress.",
    ),
    (
        "3boys, soccer, running,blue eyes, brown hair, red hair, digital art, full body,field,excited",
        "Full body digital art of three boys running on a soccer field. One with blue eyes and brown hair, one with red hair, all looking excited.",
    ),
    (
        "1girl,arisa mizuki, ponytail, black hair,green dress, sketch,portrait,sad, indoors",
        "Sketch portrait of a sad girl named arisa mizuki with a ponytail and black hair, wearing a green dress indoors.",
    ),
    (
        "1boy, cowboy shot, black hair,blue eyes,jeans,black jeans,t-shirt, white t-shirt,park,happy,jack hanma",
        "Cowboy shot of a happy boy named jack hanma with black hair and blue eyes, wearing a white t-shirt and black jeans in a park.",
    ),
    (
        "2girls,amelia taylor, green hair, twin braids, brown eyes, zoe wildly, red hair, short hair,blue eyes,full body, garden, gardening,3d,happy",
        "Full body 3d render of two girls gardening in a garden. Amelia Taylor with green hair in twin braids and brown eyes, and Zoe Wildly with short red hair and blue eyes, both looking happy.",
    ),
    (
        "no humans, forest, trees, digital art, landscape, daytime, serene, detailed",
        "Detailed digital art of a serene forest landscape during the daytime with no humans.",
    ),
    (
        "1girl, long hair, blonde hair, blue dress,portrait, oil painting, smiling, remilia scarlet",
        "Oil painting portrait of a smiling girl named remilia scarlet with long blonde hair wearing a blue dress.",
    ),
    (
        "1boy, short, black hair,green eyes, jeans,blue jeans,t-shirt, red t-shirt, indoors, reading, sketch,portrait",
        "Sketch portrait of a short boy with black hair and green eyes, wearing a red t-shirt and blue jeans, reading indoors.",
    ),
    (
        "2boys, soccer, playing, green shirt, blue shirt,blonde hair, brown hair,3d, full body, excited, park",
        "Full body 3d render of two boys playing soccer in a park. One wearing a green shirt with blonde hair, and the other wearing a blue shirt with brown hair, both excited.",
    ),
    (
        "1girl,arisa mifuku, short hair,black hair, red dress, smiling, digital art, portrait, indoors",
        "Digital art portrait of a smiling girl named arisa mifuku with short black hair wearing a red dress indoors.",
    ),
    (
        "3girls, dancing, blonde hair, black hair,red hair,green dress, blue dress,yellow dress,full body, garden, joyful, sketch",
        "Full body sketch of three girls dancing in a garden. One with blonde hair in a green dress, one with black hair in a blue dress, and one with red hair in a yellow dress, all looking joyful.",
    ),
    (
        "1boy, jason mavery, brown hair, blue eyes, shorts, green shorts,t-shirt, white t-shirt,beach, running, sketch, full body",
        "Full body sketch of a boy named jason mavery with brown hair and blue eyes, wearing a white t-shirt and green shorts, running on a beach.",
    ),
    (
        "no humans, cityscape, night, digital art, lights, buildings, detailed, urban, serene",
        "Detailed digital art of an urban cityscape at night with lights and buildings, no humans, serene atmosphere.",
    ),
    (
        "1girl, zoe wildly, red hair, curly hair, brown eyes, blue dress,portrait, traditional media, smiling, garden",
        "Traditional media portrait of a smiling girl named zoe wildly with curly red hair and brown eyes, wearing a blue dress in a garden.",
    ),
    (
        "2boys, playing, blue shirt, green shirt, black hair, blonde hair,3d, full body,park,joyful",
        "Full body 3d render of two boys playing in a park. One with black hair wearing a blue shirt, and the other with blonde hair wearing a green shirt, both joyful.",
    ),
    (
        "1girl, remilia scarlet, long hair,black hair, red dress,portrait, oil painting, serious, indoors",
        "Oil painting portrait of a serious girl named remilia scarlet with long black hair wearing a red dress indoors.",
    ),
    (
        "1boy,short, blue eyes,red hair,jeans,black jeans,t-shirt, green t-shirt,sketch,portrait, zoe wildly",
        "Sketch portrait of a short boy with blue eyes and red hair, wearing a green t-shirt and black jeans, named zoe wildly.",
    ),
    (
        "2girls, playing, yellow shirt,blue shirt,brown hair, red hair,full body, park, happy, 3d",
        "Full body 3d render of two girls playing in a park. One with brown hair wearing a yellow shirt, and the other with red hair wearing a blue shirt, both happy.",
    ),
    (
        "1boy, green hair, short hair, blue eyes, white shirt,red shorts,portrait, digital art, smiling, garden",
        "Digital art portrait of a smiling boy with short green hair and blue eyes, wearing a white shirt and red shorts in a garden.",
    ),
    (
        "3boys, soccer, field,blonde hair, black hair,red hair,green shirt, blue shirt,red shirt,full body,excited,digital art",
        "Full body digital art of three boys playing soccer on a field. One with blonde hair wearing a green shirt, one with black hair wearing a blue shirt, and one with red hair wearing a red shirt, all looking excited.",
    ),
    (
        "1girl, ponytail, black hair,blue dress, sketch,portrait, serious, indoors",
        "Sketch portrait of a serious girl with a ponytail and black hair, wearing a blue dress indoors.",
    ),
    (
        "1boy, jason mavery, brown hair, green eyes, shorts,red shorts,t-shirt, white t-shirt, beach, sketch,portrait, happy",
        "Sketch portrait of a happy boy named jason mavery with brown hair and green eyes, wearing a white t-shirt and red shorts on a beach.",
    ),
    (
        "2boys,red shirt, green shirt, blue eyes, brown hair,short hair,3d, full body,park,playing, joyful",
        "Full body 3d render of two boys playing in a park. One with blue eyes and brown hair wearing a red shirt, and the other with short hair wearing a green shirt, both joyful.",
    ),
    (
        "1girl, zoe wildly, short hair, red hair, brown eyes,blue dress, portrait, traditional media, smiling, garden",
        "Traditional media portrait of a smiling girl named zoe wildly with short red hair and brown eyes, wearing a blue dress in a garden.",
    ),
    (
        "no humans, forest, digital art, trees, day, peaceful, detailed, nature",
        "Detailed digital art of a peaceful forest during the day with no humans.",
    ),
    (
        "1girl, long hair, green hair, brown eyes,red dress,portrait, oil painting, serious, indoors",
        "Oil painting portrait of a serious girl with long green hair and brown eyes, wearing a red dress indoors.",
    ),
    (
        "1boy, short,black hair, green eyes,jeans, blue jeans,t-shirt,red t-shirt,indoors, reading, sketch, portrait, jack hanma",
        "Sketch portrait of a short boy named jack hanma with black hair and green eyes, wearing a red t-shirt and blue jeans, reading indoors.",
    ),
    (
        "2girls, soccer, playing, green shirt, blue shirt, blonde hair, brown hair, 3d, full body,park, excited",
        "Full body 3d render of two girls playing soccer in a park. One with blonde hair wearing a green shirt, and the other with brown hair wearing a blue shirt, both excited.",
    ),
]


# Function to shuffle tags
def shuffle_tags(data):
    shuffled_data = []
    for tags, description in data:
        tag_list = tags.split(',')
        random.shuffle(tag_list)
        shuffled_tags = ', '.join(tag_list)
        shuffled_data.append((shuffled_tags, description))
    return shuffled_data

data = shuffle_tags(data)

# Split dataset
dataset = TagsToDescriptionDataset(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Optimizer and scaler
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# Training function
def train(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        print(f"Step loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(dataloader)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, scaler)
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "tags_to_description_model.pth")

# Clear GPU memory
del model
del tokenizer
torch.cuda.empty_cache()

# Inference function
def generate_description(input_text):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load("tags_to_description_model.pth"))
    model.eval()

    input_encoding = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=BEAM_SIZE,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test inference (just to make sure, lol)
input_tags = "1girl, blue hair, hatsune miku, 3d, full body, short, girl, portrait, smiling"
generated_description = generate_description(input_tags)
print(f"Input: {input_tags}")
print(f"Generated description: {generated_description}")
