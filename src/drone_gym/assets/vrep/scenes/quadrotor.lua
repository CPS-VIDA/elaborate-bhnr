-- Drone States
FOLLOW_REF = 0 -- Follow REF object
PROP_CONTROL = 1 -- Manual control of propeller velocity from API
LAND = 2 -- REF_z = 0, prop_vel = {0, 0, 0, 0}

PROP_VEL_SIGNALS = {"prop1_vel", "prop2_vel", "prop3_vel", "prop4_vel"}

function reset_quad_position(inInts, inFloats, inStrings, inBuffer)
    -- inInts, inFloats and inStrings are tables
    -- inBuffer is a string

    local quadObjects = sim.getObjectsInTree(quadHandle, sim.sim_handle_all, 0)
    for i = 1, #quadObjects, 1 do
        sim.resetDynamicObject(quadObjects[i])
    end
    sim.setConfigurationTree(quadInitialConfig)

    local x, y, z = unpack(inFloats)
    sim.setObjectPosition(quadHandle, -1, {x, y, z})
    sim.setObjectPosition(targetObj, -1, {x, y, z})
    sim.setIntegerSignal("drone_state", FOLLOW_REF)

    -- Always return 3 tables and a string, e.g.:
    return {}, {}, {}, ""
end

function sysCall_init()
    -- Make sure we have version 2.4.13 or above (the particles are not supported otherwise)
    v = sim.getInt32Parameter(sim.intparam_program_version)
    if (v < 20413) then
        sim.displayDialog(
            "Warning",
            "The propeller model is only fully supported from V-REP version 2.4.13 and above.&&nThis simulation will not run as expected!",
            sim.dlgstyle_ok,
            false,
            "",
            nil,
            {0.8, 0, 0, 0, 0, 0}
        )
    end

    quadHandle = sim.getObjectHandle("Quadricopter")
    quadInitialConfig = sim.getConfigurationTree(quadHandle)

    -- Detach the ref sphere:
    targetObj = sim.getObjectHandle("Quadricopter_ref")
    sim.setObjectParent(targetObj, -1, true)
    -- Detach the goal sphere:
    goalObj = sim.getObjectHandle("Quadricopter_goal")
    sim.setObjectParent(goalObj, -1, true)

    d = sim.getObjectHandle("Quadricopter_base")
    propellerScripts = {-1, -1, -1, -1}
    for i = 1, 4, 1 do
        propellerScripts[i] = sim.getScriptHandle("Quadricopter_propeller_respondable" .. i)
    end
    heli = sim.getObjectAssociatedWithScript(sim.handle_self)

    -- Disable Particle simulations
    sim.setScriptSimulationParameter(sim.handle_tree, "fakeShadow", tostring(false))
    particlesAreVisible = sim.getScriptSimulationParameter(sim.handle_self, "particlesAreVisible")
    sim.setScriptSimulationParameter(sim.handle_tree, "particlesAreVisible", tostring(particlesAreVisible))
    simulateParticles = sim.getScriptSimulationParameter(sim.handle_self, "simulateParticles")
    sim.setScriptSimulationParameter(sim.handle_tree, "simulateParticles", tostring(simulateParticles))

    particlesTargetVelocities = {0, 0, 0, 0}

    pParam = 2
    iParam = 0
    dParam = 0
    vParam = -2

    cumul = 0
    lastE = 0
    pAlphaE = 0
    pBetaE = 0
    psp2 = 0
    psp1 = 0

    prevEuler = 0

    -- Prepare 2 floating views with the camera views:
    floorCam = sim.getObjectHandle("Quadricopter_floorCamera")
    frontCam = sim.getObjectHandle("Quadricopter_frontCamera")
    floorView = sim.floatingViewAdd(0.9, 0.9, 0.2, 0.2, 0)
    frontView = sim.floatingViewAdd(0.7, 0.9, 0.2, 0.2, 0)
    sim.adjustView(floorView, floorCam, 64)
    sim.adjustView(frontView, frontCam, 64)

    -- Set actuator signals to 0
    for i = 1, 4, 1 do
        sim.setFloatSignal(PROP_VEL_SIGNALS[i], 0.0)
        -- sim.setScriptSimulationParameter(propellerScripts[i], "particleVelocity", 0)
    end
    -- Set drone to LAND and other stuff
    -- sim.setIntegerSignal("drone_state", LAND)
    -- land_drone()
    sim.setIntegerSignal("drone_state", FOLLOW_REF)
end

function sysCall_cleanup()
    sim.floatingViewRemove(floorView)
    sim.floatingViewRemove(frontView)
end

function sysCall_actuation()
    local drone_state = sim.getIntegerSignal("drone_state")
    if drone_state == LAND then
        land_drone()
    elseif drone_state == FOLLOW_REF then
        go_to_ref()
    elseif drone_state == PROP_CONTROL then
        prop_control()
    end

    -- Send the desired motor velocities to the 4 rotors:
    for i = 1, 4, 1 do
        sim.setScriptSimulationParameter(propellerScripts[i], "particleVelocity", particlesTargetVelocities[i])
    end
end

function land_drone()
    -- In LAND, we will gradually set the particle velocity to 0
    local land_decay_rate = sim.getScriptSimulationParameter(sim.handle_self, "landDecayRate")

    -- First set the REF_z position to 0
    local rx, ry = unpack(sim.getObjectPosition(targetObj, -1))
    sim.setObjectPosition(targetObj, -1, {rx, ry, 0})

    -- Then, update the particle velocities until it reaches 0.1 and then set the velocity to 0
    for i = 1, 4, 1 do
        particlesTargetVelocities[i] = particlesTargetVelocities[i] * land_decay_rate
        if particlesTargetVelocities[i] <= 0.1 then
            particlesTargetVelocities[i] = 0
        end
    end
end

function go_to_ref()
    s = sim.getObjectSizeFactor(d)

    pos = sim.getObjectPosition(d, -1)

    -- Vertical control:
    targetPos = sim.getObjectPosition(targetObj, -1)
    pos = sim.getObjectPosition(d, -1)
    l = sim.getVelocity(heli)
    e = (targetPos[3] - pos[3])
    cumul = cumul + e
    pv = pParam * e
    thrust = 5.335 + pv + iParam * cumul + dParam * (e - lastE) + l[3] * vParam
    lastE = e

    -- Horizontal control:
    sp = sim.getObjectPosition(targetObj, d)
    m = sim.getObjectMatrix(d, -1)
    vx = {1, 0, 0}
    vx = sim.multiplyVector(m, vx)
    vy = {0, 1, 0}
    vy = sim.multiplyVector(m, vy)
    alphaE = (vy[3] - m[12])
    alphaCorr = 0.25 * alphaE + 2.1 * (alphaE - pAlphaE)
    betaE = (vx[3] - m[12])
    betaCorr = -0.25 * betaE - 2.1 * (betaE - pBetaE)
    pAlphaE = alphaE
    pBetaE = betaE
    alphaCorr = alphaCorr + sp[2] * 0.005 + 1 * (sp[2] - psp2)
    betaCorr = betaCorr - sp[1] * 0.005 - 1 * (sp[1] - psp1)
    psp2 = sp[2]
    psp1 = sp[1]

    -- Rotational control:
    euler = sim.getObjectOrientation(d, targetObj)
    rotCorr = euler[3] * 0.1 + 2 * (euler[3] - prevEuler)
    prevEuler = euler[3]

    -- Decide of the motor velocities:
    particlesTargetVelocities[1] = thrust * (1 - alphaCorr + betaCorr + rotCorr)
    particlesTargetVelocities[2] = thrust * (1 - alphaCorr - betaCorr - rotCorr)
    particlesTargetVelocities[3] = thrust * (1 + alphaCorr - betaCorr + rotCorr)
    particlesTargetVelocities[4] = thrust * (1 + alphaCorr + betaCorr - rotCorr)
end

function prop_control()
    for i = 1, 4, 1 do
        particlesTargetVelocities[i] = sim.getFloatSignal(PROP_VEL_SIGNALS[i])
    end
end
