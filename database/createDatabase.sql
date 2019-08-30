-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema aliceii
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema aliceii
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `aliceii` DEFAULT CHARACTER SET utf8 ;
USE `aliceii` ;

-- -----------------------------------------------------
-- Table `aliceii`.`newspaper`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `aliceii`.`newspaper` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `title` VARCHAR(255) NOT NULL,
  `content` LONGTEXT NOT NULL,
  `label` INT(11) NULL DEFAULT NULL,
  `sentences` INT(11) NULL DEFAULT NULL,
  `key_sentence` VARCHAR(255) NULL DEFAULT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 1090
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `aliceii`.`violentCriteria`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `aliceii`.`violentCriteria` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `violentWord` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `aliceii`.`sentences`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `aliceii`.`sentences` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `newspaper_id` INT NOT NULL,
  `content` MEDIUMTEXT NOT NULL,
  `label` INT NULL,
  `isTrain` INT NULL,
  `RandomForest_tfidf_label` INT NULL,
  `RandomForest_word2vec_label` INT NULL,
  `LogisticRegression_tfidf_label` INT NULL,
  `LogisticRegression_word2vec_label` INT NULL,
  `Gaussian_tfidf_label` INT NULL,
  `Gaussian_word2vec_label` INT NULL
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

CREATE TABLE IF NOT EXISTS `aliceii`.`tmp` (
  `id` INT NOT NULL,
  `label` INT NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = big5;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
