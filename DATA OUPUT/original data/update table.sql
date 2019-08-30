CREATE TABLE IF NOT EXISTS `aliceii`.`sentences` (
  `id` INT NOT NULL AUTO_INCREMENT primary key,
  `newspaper_id` INT NOT NULL,
  `content` MEDIUMTEXT NOT NULL,
  `label` INT NULL
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;