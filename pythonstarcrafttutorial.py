# Other Imports
import random
import cv2
import numpy as np
import time

# SC2 Imports
import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer, Human
from sc2.constants import *
#from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
#CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY, \
#ROBOTICSBAY, WARPGATE, COLOSSUS, IMMORTAL, RESEARCH_WARPGATE

HEADLESS = False

class RobertBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 80
        self.do_something_after = 0
        self.train_data = []

        self.build_observer_after = 0

        self.MAX_GATEWAYS = 4 # per base
        self.MAX_ROBOS = 2
        self.MAX_STARGATES = 2
        self.late_game = False

        self.warpgate_started = False
        self.thermal_lance_started = False


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))
        
    async def on_step(self, iteration):
        self.iteration = iteration

        if not self.warpgate_started:
            await self.start_warpgate()

        if not self.thermal_lance_started:
            await self.start_thermal_lance()

        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.expand()
        await self.build_assimilators()
        await self.build_offensive_force()
        await self.offensive_force_buildings()
        await self.intel()
        await self.attack()

    async def start_warpgate(self):
        if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(RESEARCH_WARPGATE) and not self.warpgate_started:
            ccore = self.units(CYBERNETICSCORE).ready.first
            await self.do(ccore(RESEARCH_WARPGATE))
            self.warpgate_started = True
    
    async def start_thermal_lance(self):
        if self.units(ROBOTICSBAY).ready.exists and self.can_afford(RESEARCH_EXTENDEDTHERMALLANCE) and not self.thermal_lance_started:
            robo_bay = self.units(ROBOTICSBAY).ready.first
            await self.do(robo_bay(RESEARCH_EXTENDEDTHERMALLANCE))
            self.thermal_lance_started = True

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8) # width by height
        # UNIT: [SIZE, (BGR COLOR)]
        draw_dict = {
            # UNITS
            PROBE: [1, (55, 200, 0)],
            OBSERVER: [1, (255, 255, 255)],
            STALKER: [3, (255, 200, 0)],
            VOIDRAY: [3, (255, 100, 0)],
            IMMORTAL: [3, (0, 100, 255)],
            COLOSSUS: [3, (0, 200, 255)],

            # BUILDINGS
            NEXUS: [10, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            ASSIMILATOR: [2, (55, 200, 0)],
            GATEWAY: [5, (200, 100, 0)],
            WARPGATE: [5, (200, 100, 50)],
            CYBERNETICSCORE: [5, (150, 150, 0)],
            STARGATE: [5, (255, 0, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],
            ROBOTICSBAY: [5, (215, 155, 100)]
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        # ENEMY BUILDINGS
        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)

        # ENEMY UNITS
        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / (self.supply_cap + .00001)
        if population_ratio > 1.0:
            population_ratio = 1.0
        
        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap - self.supply_left + .00001)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3) # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        
        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))

        else:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0 and self.iteration > self.build_observer_after:
                    await self.do(rf.train(OBSERVER))
                    build_observer_after = self.iteration + self.ITERATIONS_PER_MINUTE
    
    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    async def build_workers(self):
        if self.units(PROBE).amount <= self.MAX_WORKERS and self.units(PROBE).amount < (self.units(NEXUS).amount + 0.5) * 22:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 * self.units(NEXUS).amount and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)
                
    async def build_assimilators(self):
        if self.units(GATEWAY).amount > 0 or self.units(WARPGATE).amount > 0:
            for nexus in self.units(NEXUS).ready:
                geysers = self.state.vespene_geyser.closer_than(10.0, nexus)
                for geyser in geysers:
                    if not self.can_afford(ASSIMILATOR):
                        break
                    worker = self.select_build_worker(geyser.position)
                    if worker is None:
                        break
                    if not self.units(ASSIMILATOR).closer_than(1.0, geyser).exists:
                        await self.do(worker.build(ASSIMILATOR, geyser))

    async def expand(self):
        if self.can_afford(NEXUS) and (self.units(PROBE).amount > self.units(NEXUS).amount * 16 or self.units(PROBE).amount >= self.MAX_WORKERS * .9):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE) and self.units(NEXUS).amount >= 2:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif self.units(CYBERNETICSCORE).ready.exists and self.units(NEXUS).amount >= 2 and self.units(ROBOTICSFACILITY).amount < self.MAX_ROBOS and self.units(ROBOTICSFACILITY).amount + self.units(STARGATE).amount < self.units(NEXUS).amount:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)
            elif self.units(ROBOTICSFACILITY).ready.amount > 0 and self.units(NEXUS).amount >= 2 and self.units(ROBOTICSBAY).amount == 0:
                if self.can_afford(ROBOTICSBAY) and not self.already_pending(ROBOTICSBAY):
                    await self.build(ROBOTICSBAY, near=pylon)
            elif (self.units(WARPGATE).amount < self.MAX_GATEWAYS * self.units(NEXUS).amount and self.units(NEXUS).amount >= 2) or (self.units(NEXUS).amount < 2 and self.units(WARPGATE).amount < self.MAX_GATEWAYS / 2):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            elif self.units(CYBERNETICSCORE) and self.units(NEXUS).amount >= 2 and self.units(STARGATE).amount < self.MAX_STARGATES and self.units(ROBOTICSFACILITY).amount + self.units(STARGATE).amount < self.units(NEXUS).amount:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

            

    async def build_offensive_force(self):
        if self.warpgate_started:
            for gateway in self.units(GATEWAY).ready.noqueue:
                if self.units(CYBERNETICSCORE).ready:
                    if self.can_afford(STALKER) and self.supply_left >= 2:
                        await self.do(gateway.train(STALKER))
        
        # building_from_robo = False
        # if self.units(ROBOTICSFACILITY).amount == 0:
        #     building_from_robo = True

        # Building Colossus and Immortals
        for robo in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.units(ROBOTICSBAY).amount > 0:
                if self.can_afford(COLOSSUS) and self.supply_left >= 6:
                    await self.do(robo.train(COLOSSUS))
                    building_from_robo = True
            elif self.can_afford(IMMORTAL) and self.supply_left >= 4:
                await self.do(robo.train(IMMORTAL))
                building_from_robo = True

        # building_from_stargate = False
        # if self.units(STARGATE).amount == 0:
        #     building_from_stargate = True

        # Building Void Rays only if already producing out of Robo
        # if building_from_robo:
        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left >= 4:
                await self.do(stargate.train(VOIDRAY))
                building_from_stargate = True

        # if building_from_robo and building_from_stargate:
            # Building Stalkers if all stargates and robos are already in use
        closest_nexus = self.units(NEXUS).closest_to(self.enemy_start_locations[0])
        if self.units(PYLON).ready.amount > 0:
            pylon = self.units(PYLON).ready.closest_to(closest_nexus)
        # if wargate finished
        for warpgate in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            if AbilityId.WARPGATETRAIN_STALKER in abilities and self.units(CYBERNETICSCORE).ready.amount > 0 and self.can_afford(STALKER) and self.supply_left >= 2:
                pos = pylon.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    #return ActionResult.CantFindPlacementLocation
                    print("can't place")
                    return
                await self.do(warpgate.warp_in(STALKER, placement))


    def find_target(self, state):
      #  if len(self.known_enemy_units) > 0:
      #      return random.choice(self.known_enemy_units)
      #  elif len(self.known_enemy_structures) > 0:
      #      return random.choice(self.known_enemy_structures)
      #  else:
        return self.enemy_start_locations[0]

    async def attack(self):
        if len(self.units(VOIDRAY).idle) + len(self.units(COLOSSUS).idle) + len(self.units(STALKER).idle) + len(self.units(IMMORTAL).idle) > 0:
            option = random.randrange(0, 5)
            target = False
            if self.iteration > self.do_something_after:
                
                closest_nexus = self.units(NEXUS).closest_to(self.enemy_start_locations[0])
                
                if option == 0:
                    # wait 1/2 of a second
                    wait = 165/2
                    self.do_something_after = self.iteration + wait

                elif option == 1:
                    # attack enemy unit closest to nexus closest to enemy base
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(closest_nexus)

                elif option == 2:
                    # attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = self.known_enemy_structures.closest_to(closest_nexus)

                elif option == 3:
                    # attack enemy start location
                    target = self.enemy_start_locations[0]

                elif option == 4:
                    # reposition to closest nexus
                    target = closest_nexus.position

                if target:
                    for voidray in self.units(VOIDRAY).idle:
                        await self.do(voidray.attack(target))
                    for colossus in self.units(COLOSSUS).idle:
                        await self.do(colossus.attack(target))
                    for immortal in self.units(IMMORTAL).idle:
                        await self.do(immortal.attack(target))
                    for stalker in self.units(STALKER).idle:
                        await self.do(stalker.attack(target))
                        
                y = np.zeros(5)
                y[option] = 1
                print(y)
                self.train_data.append([y,self.flipped])


    async def attack_with(self, unit):
        if self.units(unit).amount > 5 or self.late_game:
            for u in self.units(unit).idle:
                self.late_game = True
                await self.do(u.attack(self.find_target(self.state)))

        # elif self.units(unit).amount > 2 and not self.late_game:
        #     if len(self.known_enemy_units) > 0:
        #         for u in self.units(unit).idle:
        #             await self.do(u.attack(random.choice(self.known_enemy_units)))

def main():
    i = 0
    while i < 1:
        run_game(maps.get("AutomatonLE"), [
            Bot(Race.Protoss, RobertBot(), name="RobbyT"),
            Computer(Race.Zerg, Difficulty.Hard),
        ], realtime=False)
        i = i + 1

if __name__ == '__main__':
    main()